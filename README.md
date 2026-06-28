# RGM MCP Server

A Python MCP server exposing analytical tools for **Revenue Growth Management (RGM)** in the
Alco-Bev / North America space. Designed to work with flat-file exports from
[NIQ / NielsenIQ](https://nielseniq.com) (CSV or Parquet).

## Tools

| Tool | Description |
|---|---|
| `get_nielsen_input_schema` | Show the required and optional column schema for each NIQ input file before you start |
| `build_analytical_base_table` | Join NIQ sales + distribution + pricing exports into a clean, model-ready Analytical Base Table (ABT) with log-transformed columns |
| `calculate_price_elasticity` | OLS log-log regression → own-price & cross-price elasticity per segment (market, channel, SKU, etc.) |
| `score_promo_effectiveness` | Rolling-baseline lift %, incremental volume/revenue, and trade ROI per promo event |
| `recommend_dynamic_pricing` | Grid-search over ±N% price moves to find the revenue-maximising price given a margin floor and elasticity |
| `optimize_promo_calendar` | Greedy ROI-ranked promo event scheduling within total budget + max-events-per-SKU constraints |
| `compute_competitive_price_index` | Volume-weighted Competitive Price Index (own price / competitor price × 100) by category / brand / pack size / market |

## Requirements

- Python 3.10+
- Dependencies: `fastmcp`, `pandas`, `pyarrow`, `numpy`, `scipy`

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.\.venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Running locally (stdio — for use with Bob / Claude Desktop)

```bash
python src/server.py
```

Register in your MCP client config (`mcp.json`):

```json
{
  "mcpServers": {
    "rgm-mcp-server": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/src/server.py"]
    }
  }
}
```

## Typical workflow

0. **Check the input schema** for your NIQ files:
   > "What columns do I need in my NIQ sales file?"

1. **Build the ABT** from your NIQ exports:
   > "Build me the analytical base table from sales.csv, dist.csv, and pricing.csv, save to abt.csv"

2. **Compute price elasticities** by market:
   > "What are the price elasticities by market from abt.csv?"

3. **Score past promos**:
   > "Score promo effectiveness by market from abt.csv"

4. **Get pricing recommendations** (uses elasticity output):
   > "Recommend prices for Chicago and New York given a 30% margin floor and COGS of $5"

5. **Build the promo calendar**:
   > "Optimise a promo calendar for H2 2025 with a $200k budget"

6. **Competitive price indexing**:
   > "Compute the competitive price index for BrandA across all markets"

## NIQ input file schemas

Call `get_nielsen_input_schema()` at any time to get the full column spec. Quick reference:

### sales file (`sales_file`)

| Column | Type | Required | Description |
|---|---|---|---|
| `period_end_date` | date | ✅ | Week- or month-ending date (YYYY-MM-DD) |
| `upc` | string | ✅ | SKU / Universal Product Code |
| `market` | string | ✅ | NIQ retail geography |
| `channel` | string | ✅ | Trade channel (Grocery, Liquor, Club, etc.) |
| `unit_sales` | numeric | ✅ | Units sold in the period |
| `dollar_sales` | numeric | ✅ | Dollar revenue in the period |
| `avg_price_per_unit` | numeric | ✅ | Average shelf price (USD, no $ symbol) |
| `brand` | string | — | Brand name (required for CPI tool) |
| `category` | string | — | Category / sub-category |
| `pack_size` | string | — | Pack size / volume format |
| `any_promo_flag` | 0 / 1 | — | 1 = promoted week (required for promo tools) |
| `trade_spend` | numeric | — | Trade spend in USD (required for promo ROI) |

### distribution file (`distribution_file`)

| Column | Type | Required | Description |
|---|---|---|---|
| `period_end_date` | date | ✅ | Must match sales file exactly |
| `upc` | string | ✅ | Must match sales file exactly |
| `market` | string | ✅ | Must match sales file exactly |
| `channel` | string | ✅ | Must match sales file exactly |
| `total_distribution_points` | numeric | ✅ | NIQ TDP (% ACV weighted distribution) |

### pricing file (`pricing_file`)

| Column | Type | Required | Description |
|---|---|---|---|
| `period_end_date` | date | ✅ | Must match sales file exactly |
| `upc` | string | ✅ | Must match sales file exactly |
| `market` | string | ✅ | Must match sales file exactly |
| `channel` | string | ✅ | Must match sales file exactly |
| `competitor_brand` | string | ✅ | Competitor brand name |
| `comp_avg_price` | numeric | ✅ | Competitor average shelf price (USD) |
| `avg_price_per_unit` | numeric | — | Own price (can be omitted if in sales file) |

> All three files are joined on `period_end_date + upc + market + channel`. Values must match exactly (case-sensitive) across files.
>
> All column names can be overridden via tool parameters when calling each tool.

## License

MIT
