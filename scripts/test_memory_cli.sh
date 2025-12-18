#!/bin/bash
# =============================================================================
# ContextFS Memory Operations Test Script
# Tests all memory operations via CLI
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Use python module directly
CTX="python -m contextfs.cli"

echo -e "${BLUE}=== ContextFS Memory Operations Test ===${NC}\n"

# Helper to extract ID from output
extract_id() {
    echo "$1" | grep -oE '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' | head -1
}

# =============================================================================
echo -e "${YELLOW}1. Status Check${NC}"
# =============================================================================
$CTX status
echo ""

# =============================================================================
echo -e "${YELLOW}2. Basic Save Operations${NC}"
# =============================================================================

echo "Saving fact..."
OUTPUT=$($CTX save "Test API endpoint is /api/v1/test" --type fact --tags "api,test" --summary "Test endpoint")
echo "$OUTPUT"
FACT_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Saved fact: ${FACT_ID:0:12}...${NC}"

echo -e "\nSaving decision..."
OUTPUT=$($CTX save "Decided to use SQLite for local storage" --type decision --tags "database,architecture")
echo "$OUTPUT"
DECISION_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Saved decision: ${DECISION_ID:0:12}...${NC}"

echo -e "\nSaving procedural..."
OUTPUT=$($CTX save "To deploy: run ./deploy.sh then verify health endpoint" --type procedural --tags "deployment,ops")
echo "$OUTPUT"
PROC_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Saved procedural: ${PROC_ID:0:12}...${NC}"

# =============================================================================
echo -e "\n${YELLOW}3. List and Search${NC}"
# =============================================================================

echo "Listing recent memories..."
$CTX list --limit 5

echo -e "\nSearching for 'API'..."
$CTX search "API endpoint" --limit 3

# =============================================================================
echo -e "\n${YELLOW}4. Recall by ID${NC}"
# =============================================================================

echo "Recalling fact by partial ID..."
$CTX recall "${FACT_ID:0:8}"

# =============================================================================
echo -e "\n${YELLOW}5. Evolve Operations (Lineage)${NC}"
# =============================================================================

echo "Creating memory to evolve..."
OUTPUT=$($CTX save "Rate limit is 100 requests/minute" --type fact --tags "api,limits" --summary "Rate limit v1")
echo "$OUTPUT"
V1_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Created v1: ${V1_ID:0:12}...${NC}"

echo -e "\nEvolving to v2..."
OUTPUT=$($CTX evolve "${V1_ID:0:8}" "Rate limit increased to 500 requests/minute" --summary "Rate limit v2")
echo "$OUTPUT"
V2_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Created v2: ${V2_ID:0:12}...${NC}"

echo -e "\nEvolving to v3..."
OUTPUT=$($CTX evolve "${V2_ID:0:8}" "Rate limit is 1000 requests/minute for authenticated users" --summary "Rate limit v3" --tags "premium")
echo "$OUTPUT"
V3_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Created v3: ${V3_ID:0:12}...${NC}"

# =============================================================================
echo -e "\n${YELLOW}6. View Lineage${NC}"
# =============================================================================

echo "Getting lineage for v3..."
$CTX lineage "${V3_ID:0:8}"

echo -e "\nGetting ancestors only..."
$CTX lineage "${V3_ID:0:8}" --direction ancestors

# =============================================================================
echo -e "\n${YELLOW}7. Link Operations${NC}"
# =============================================================================

echo "Creating memories to link..."
OUTPUT=$($CTX save "Auth service handles JWT tokens" --type fact --tags "auth,service")
AUTH_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Created auth memory: ${AUTH_ID:0:12}...${NC}"

OUTPUT=$($CTX save "Session store uses Redis" --type fact --tags "session,redis")
SESSION_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Created session memory: ${SESSION_ID:0:12}...${NC}"

echo -e "\nLinking: auth -> session (related_to)..."
$CTX link "${AUTH_ID:0:8}" "${SESSION_ID:0:8}" related_to
echo -e "${GREEN}Link created${NC}"

echo -e "\nLinking: v3 rate limit -> auth (references)..."
$CTX link "${V3_ID:0:8}" "${AUTH_ID:0:8}" references
echo -e "${GREEN}Link created${NC}"

# =============================================================================
echo -e "\n${YELLOW}8. Find Related Memories${NC}"
# =============================================================================

echo "Finding memories related to auth..."
$CTX related "${AUTH_ID:0:8}"

echo -e "\nFinding with depth 2..."
$CTX related "${V3_ID:0:8}" --depth 2

# =============================================================================
echo -e "\n${YELLOW}9. Merge Operations${NC}"
# =============================================================================

echo "Creating memories to merge..."
OUTPUT=$($CTX save "Frontend framework is React 18" --type fact --tags "frontend,react")
REACT_ID=$(extract_id "$OUTPUT")

OUTPUT=$($CTX save "Frontend uses TypeScript 5.0" --type fact --tags "frontend,typescript")
TS_ID=$(extract_id "$OUTPUT")

OUTPUT=$($CTX save "Build tool is Vite 5" --type fact --tags "frontend,build")
VITE_ID=$(extract_id "$OUTPUT")

echo -e "${GREEN}Created 3 frontend memories${NC}"

echo -e "\nMerging with union strategy..."
OUTPUT=$($CTX merge "${REACT_ID:0:8}" "${TS_ID:0:8}" "${VITE_ID:0:8}" --summary "Frontend tech stack" --strategy union)
echo "$OUTPUT"
MERGED_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Merged into: ${MERGED_ID:0:12}...${NC}"

echo -e "\nViewing merge lineage..."
$CTX lineage "${MERGED_ID:0:8}"

# =============================================================================
echo -e "\n${YELLOW}10. Split Operations${NC}"
# =============================================================================

echo "Creating memory to split..."
OUTPUT=$($CTX save "Environment: DEBUG=true, LOG_LEVEL=info, MAX_CONN=100" --type fact --tags "config,env")
CONFIG_ID=$(extract_id "$OUTPUT")
echo -e "${GREEN}Created config memory: ${CONFIG_ID:0:12}...${NC}"

echo -e "\nSplitting into 3 parts..."
$CTX split "${CONFIG_ID:0:8}" "DEBUG=true" "LOG_LEVEL=info" "MAX_CONN=100" --summaries "Debug flag|Log level|Max connections"

# =============================================================================
echo -e "\n${YELLOW}11. Graph Status${NC}"
# =============================================================================

$CTX graph-status

# =============================================================================
echo -e "\n${GREEN}=== All Tests Completed Successfully ===${NC}"
# =============================================================================

echo -e "\nSummary of created memories:"
echo "  - Fact: ${FACT_ID:0:12}..."
echo "  - Decision: ${DECISION_ID:0:12}..."
echo "  - Procedural: ${PROC_ID:0:12}..."
echo "  - Rate limit lineage: ${V1_ID:0:12} -> ${V2_ID:0:12} -> ${V3_ID:0:12}"
echo "  - Auth: ${AUTH_ID:0:12}..."
echo "  - Session: ${SESSION_ID:0:12}..."
echo "  - Merged frontend: ${MERGED_ID:0:12}..."
echo "  - Split config: ${CONFIG_ID:0:12}..."
