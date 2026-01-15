"""Team management API routes for ContextFS.

Provides endpoints for:
- Creating and managing teams (Team tier only)
- Inviting and removing team members
- Managing team roles
- Team memory visibility
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth.api_keys import APIKey, User
from service.api.auth_middleware import require_auth
from service.db.models import (
    SubscriptionModel,
    TeamInvitationModel,
    TeamMemberModel,
    TeamModel,
    UserModel,
)
from service.db.session import get_session_dependency

router = APIRouter(prefix="/api/teams", tags=["teams"])


# =============================================================================
# Pydantic Models
# =============================================================================


class CreateTeamRequest(BaseModel):
    """Request to create a new team."""

    name: str
    description: str | None = None


class TeamResponse(BaseModel):
    """Team info response."""

    id: str
    name: str
    description: str | None
    owner_id: str
    member_count: int
    created_at: str


class TeamMemberResponse(BaseModel):
    """Team member info."""

    user_id: str
    email: str
    name: str | None
    role: str
    joined_at: str


class InviteMemberRequest(BaseModel):
    """Request to invite a member."""

    email: str
    role: str = "member"  # member or admin


class InviteMemberResponse(BaseModel):
    """Response after sending invitation."""

    invitation_id: str
    email: str
    role: str
    expires_at: str


class UpdateMemberRoleRequest(BaseModel):
    """Request to update member role."""

    role: str  # member, admin


class AcceptInvitationRequest(BaseModel):
    """Request to accept an invitation."""

    token: str


# =============================================================================
# Tier Limits
# =============================================================================

TIER_LIMITS = {
    # Free: Cloud sync enabled for evaluation, limited capacity
    "free": {
        "can_create_team": False,
        "cloud_sync": True,  # Enabled for evaluation
        "device_limit": 2,
        "memory_limit": 5000,
    },
    # Pro: Power user, more capacity, still solo
    "pro": {
        "can_create_team": False,
        "cloud_sync": True,
        "device_limit": 5,
        "memory_limit": 50000,
    },
    # Team: Collaboration features
    "team": {
        "can_create_team": True,
        "cloud_sync": True,
        "device_limit": 10,  # per user
        "memory_limit": -1,  # unlimited
        "seats_included": 5,
    },
    # Enterprise: Self-hosted, unlimited
    "enterprise": {
        "can_create_team": True,
        "cloud_sync": True,
        "device_limit": -1,
        "memory_limit": -1,
        "seats_included": -1,
    },
    # Admin: System administrators, unlimited
    "admin": {
        "can_create_team": True,
        "cloud_sync": True,
        "device_limit": -1,
        "memory_limit": -1,
        "seats_included": -1,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_user_subscription(session: AsyncSession, user_id: str) -> SubscriptionModel | None:
    """Get user's subscription."""
    result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def _get_user_teams(session: AsyncSession, user_id: str) -> list[TeamModel]:
    """Get all teams a user belongs to."""
    result = await session.execute(
        select(TeamModel)
        .join(TeamMemberModel, TeamModel.id == TeamMemberModel.team_id)
        .where(TeamMemberModel.user_id == user_id)
    )
    return list(result.scalars().all())


async def _get_team_member_count(session: AsyncSession, team_id: str) -> int:
    """Get count of team members."""
    result = await session.execute(
        select(TeamMemberModel).where(TeamMemberModel.team_id == team_id)
    )
    return len(result.scalars().all())


async def _is_team_admin(session: AsyncSession, team_id: str, user_id: str) -> bool:
    """Check if user is team owner or admin."""
    result = await session.execute(
        select(TeamMemberModel).where(
            TeamMemberModel.team_id == team_id,
            TeamMemberModel.user_id == user_id,
            TeamMemberModel.role.in_(["owner", "admin"]),
        )
    )
    return result.scalar_one_or_none() is not None


async def _is_team_member(session: AsyncSession, team_id: str, user_id: str) -> bool:
    """Check if user is a team member."""
    result = await session.execute(
        select(TeamMemberModel).where(
            TeamMemberModel.team_id == team_id,
            TeamMemberModel.user_id == user_id,
        )
    )
    return result.scalar_one_or_none() is not None


# =============================================================================
# Team Routes
# =============================================================================


async def _is_admin_user(session: AsyncSession, user_id: str) -> bool:
    """Check if user is an admin via database flag."""
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    db_user = result.scalar_one_or_none()
    return db_user is not None and getattr(db_user, "is_admin", False)


@router.post("", response_model=TeamResponse)
async def create_team(
    request: CreateTeamRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Create a new team (Team tier required, admins bypass)."""
    user, _ = auth

    # Check subscription tier (admins bypass this check)
    subscription = await _get_user_subscription(session, user.id)
    tier = subscription.tier if subscription else "free"

    is_admin = await _is_admin_user(session, user.id)
    if not is_admin and not TIER_LIMITS.get(tier, {}).get("can_create_team", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Team creation requires Team or Enterprise tier (current: {tier})",
        )

    # Check if user already owns a team
    existing_teams = await _get_user_teams(session, user.id)
    owned_teams = [t for t in existing_teams if t.owner_id == user.id]
    if owned_teams:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already own a team. Currently limited to one team per user.",
        )

    # Create team
    team_id = str(uuid4())
    team = TeamModel(
        id=team_id,
        name=request.name,
        description=request.description,
        owner_id=user.id,
    )
    session.add(team)

    # Flush to create team in DB before adding members (foreign key)
    await session.flush()

    # Add owner as team member with 'owner' role
    member = TeamMemberModel(
        team_id=team_id,
        user_id=user.id,
        role="owner",
        invited_by=user.id,
    )
    session.add(member)

    # Update subscription with team_id
    if subscription:
        subscription.team_id = team_id
        subscription.seats_included = TIER_LIMITS[tier].get("seats_included", 5)
        subscription.seats_used = 1

    await session.commit()

    return TeamResponse(
        id=team_id,
        name=team.name,
        description=team.description,
        owner_id=team.owner_id,
        member_count=1,
        created_at=team.created_at.isoformat(),
    )


class TeamsListResponse(BaseModel):
    """Response for listing teams."""

    teams: list[TeamResponse]


@router.get("", response_model=TeamsListResponse)
async def list_teams(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """List teams the user belongs to."""
    user, _ = auth

    teams = await _get_user_teams(session, user.id)

    result = []
    for team in teams:
        member_count = await _get_team_member_count(session, team.id)
        result.append(
            TeamResponse(
                id=team.id,
                name=team.name,
                description=team.description,
                owner_id=team.owner_id,
                member_count=member_count,
                created_at=team.created_at.isoformat(),
            )
        )

    return TeamsListResponse(teams=result)


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get team details."""
    user, _ = auth

    # Verify membership
    if not await _is_team_member(session, team_id, user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    result = await session.execute(select(TeamModel).where(TeamModel.id == team_id))
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    member_count = await _get_team_member_count(session, team_id)

    return TeamResponse(
        id=team.id,
        name=team.name,
        description=team.description,
        owner_id=team.owner_id,
        member_count=member_count,
        created_at=team.created_at.isoformat(),
    )


class TeamMembersListResponse(BaseModel):
    """Response for listing team members."""

    members: list[TeamMemberResponse]


@router.get("/{team_id}/members", response_model=TeamMembersListResponse)
async def list_team_members(
    team_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """List team members."""
    user, _ = auth

    # Verify membership
    if not await _is_team_member(session, team_id, user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    result = await session.execute(
        select(TeamMemberModel, UserModel)
        .join(UserModel, TeamMemberModel.user_id == UserModel.id)
        .where(TeamMemberModel.team_id == team_id)
    )

    members = []
    for member, member_user in result.all():
        members.append(
            TeamMemberResponse(
                user_id=member.user_id,
                email=member_user.email,
                name=member_user.name,
                role=member.role,
                joined_at=member.joined_at.isoformat(),
            )
        )

    return TeamMembersListResponse(members=members)


@router.post("/{team_id}/invite", response_model=InviteMemberResponse)
async def invite_member(
    team_id: str,
    request: InviteMemberRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Invite a user to the team (admin only)."""
    user, _ = auth

    # Verify admin permissions
    if not await _is_team_admin(session, team_id, user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only team owners and admins can invite members",
        )

    # Check seat limit
    result = await session.execute(select(TeamModel).where(TeamModel.id == team_id))
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    # Get team owner's subscription for seat limits
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.team_id == team_id)
    )
    subscription = sub_result.scalar_one_or_none()

    if subscription and subscription.seats_included > 0:
        member_count = await _get_team_member_count(session, team_id)
        if member_count >= subscription.seats_included:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Team seat limit reached ({subscription.seats_included} seats)",
            )

    # Check if already a member
    existing_user = await session.execute(select(UserModel).where(UserModel.email == request.email))
    existing = existing_user.scalar_one_or_none()
    if existing and await _is_team_member(session, team_id, existing.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a team member",
        )

    # Check for pending invitation
    existing_invite = await session.execute(
        select(TeamInvitationModel).where(
            TeamInvitationModel.team_id == team_id,
            TeamInvitationModel.email == request.email,
            TeamInvitationModel.accepted_at.is_(None),
            TeamInvitationModel.expires_at > datetime.now(timezone.utc),
        )
    )
    if existing_invite.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation already pending for this email",
        )

    # Create invitation
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    invitation = TeamInvitationModel(
        id=str(uuid4()),
        team_id=team_id,
        email=request.email,
        role=request.role,
        invited_by=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    session.add(invitation)
    await session.commit()

    # TODO: Send invitation email with token

    return InviteMemberResponse(
        invitation_id=invitation.id,
        email=request.email,
        role=request.role,
        expires_at=expires_at.isoformat(),
    )


@router.post("/invitations/accept")
async def accept_invitation(
    request: AcceptInvitationRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Accept a team invitation."""
    user, _ = auth

    token_hash = hashlib.sha256(request.token.encode()).hexdigest()

    result = await session.execute(
        select(TeamInvitationModel).where(
            TeamInvitationModel.token_hash == token_hash,
            TeamInvitationModel.email == user.email,
            TeamInvitationModel.accepted_at.is_(None),
            TeamInvitationModel.expires_at > datetime.now(timezone.utc),
        )
    )
    invitation = result.scalar_one_or_none()

    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invitation",
        )

    # Add user to team
    member = TeamMemberModel(
        team_id=invitation.team_id,
        user_id=user.id,
        role=invitation.role,
        invited_by=invitation.invited_by,
    )
    session.add(member)

    # Mark invitation as accepted
    invitation.accepted_at = datetime.now(timezone.utc)

    # Update seat count
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.team_id == invitation.team_id)
    )
    subscription = sub_result.scalar_one_or_none()
    if subscription:
        subscription.seats_used += 1

    await session.commit()

    return {"status": "joined", "team_id": invitation.team_id}


class PendingInvitationResponse(BaseModel):
    """Pending invitation info for the current user."""

    id: str
    team_id: str
    team_name: str
    email: str
    role: str
    invited_by_email: str
    expires_at: str
    created_at: str


@router.get("/invitations/pending", response_model=list[PendingInvitationResponse])
async def get_pending_invitations(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get pending invitations for the current user."""
    user, _ = auth

    result = await session.execute(
        select(TeamInvitationModel, TeamModel, UserModel)
        .join(TeamModel, TeamInvitationModel.team_id == TeamModel.id)
        .join(UserModel, TeamInvitationModel.invited_by == UserModel.id)
        .where(
            TeamInvitationModel.email == user.email,
            TeamInvitationModel.accepted_at.is_(None),
            TeamInvitationModel.expires_at > datetime.now(timezone.utc),
        )
    )

    invitations = []
    for invitation, team, invited_by_user in result.all():
        invitations.append(
            PendingInvitationResponse(
                id=invitation.id,
                team_id=invitation.team_id,
                team_name=team.name,
                email=invitation.email,
                role=invitation.role,
                invited_by_email=invited_by_user.email,
                expires_at=invitation.expires_at.isoformat(),
                created_at=invitation.created_at.isoformat(),
            )
        )

    return invitations


@router.put("/{team_id}/members/{user_id}/role")
async def update_member_role(
    team_id: str,
    user_id: str,
    request: UpdateMemberRoleRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Update a team member's role (admin only)."""
    current_user, _ = auth

    # Verify admin permissions
    if not await _is_team_admin(session, team_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only team owners and admins can update roles",
        )

    # Can't change owner role
    result = await session.execute(
        select(TeamMemberModel).where(
            TeamMemberModel.team_id == team_id,
            TeamMemberModel.user_id == user_id,
        )
    )
    member = result.scalar_one_or_none()

    if not member:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    if member.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change owner role",
        )

    # Validate role
    if request.role not in ["member", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'member' or 'admin'",
        )

    member.role = request.role
    await session.commit()

    return {"status": "updated", "role": request.role}


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    team_id: str,
    user_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Remove a member from the team (admin only, or self)."""
    current_user, _ = auth

    # Can remove self or must be admin
    is_self = user_id == current_user.id
    is_admin = await _is_team_admin(session, team_id, current_user.id)

    if not is_self and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only team admins can remove other members",
        )

    # Can't remove owner
    result = await session.execute(
        select(TeamMemberModel).where(
            TeamMemberModel.team_id == team_id,
            TeamMemberModel.user_id == user_id,
        )
    )
    member = result.scalar_one_or_none()

    if not member:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    if member.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove team owner",
        )

    await session.execute(
        delete(TeamMemberModel).where(
            TeamMemberModel.team_id == team_id,
            TeamMemberModel.user_id == user_id,
        )
    )

    # Update seat count
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.team_id == team_id)
    )
    subscription = sub_result.scalar_one_or_none()
    if subscription and subscription.seats_used > 0:
        subscription.seats_used -= 1

    await session.commit()

    return {"status": "removed"}


@router.delete("/{team_id}")
async def delete_team(
    team_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Delete a team (owner only)."""
    user, _ = auth

    result = await session.execute(select(TeamModel).where(TeamModel.id == team_id))
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    if team.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only team owner can delete the team",
        )

    # Clear team_id from subscription BEFORE deleting team (foreign key constraint)
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.team_id == team_id)
    )
    subscription = sub_result.scalar_one_or_none()
    if subscription:
        subscription.team_id = None
        subscription.seats_used = 1
        await session.flush()

    # Delete team (cascade will delete members and invitations)
    await session.execute(delete(TeamModel).where(TeamModel.id == team_id))

    await session.commit()

    return {"status": "deleted"}
