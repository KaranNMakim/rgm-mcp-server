# RGM MCP Server

A Python MCP server exposing analytical tools for **Revenue Growth Management (RGM)** in the
Alco-Bev / North America space. Designed to work with flat-file exports from
[NIQ / NielsenIQ](https://nielseniq.com) (CSV or Parquet).

## Tools

| Tool | Description |
|---|---|
| `build_analytical_base_table` | Join NIQ sales + distribution + pricing exports into a clean, model-ready Analytical Base Table (ABT) with log-transformed columns |
| `calculate_price_elasticity` | OLS log-log regression → own-price & cross-price elasticity per segment (market, channel, SKU, etc.) |
| `score_promo_effectiveness` | Rolling-baseline lift %, incremental volume/revenue, and trade ROI per promo event |
| `recommend_dynamic_pricing` | Grid-search over ±N% price moves to find the revenue-maximising price given a margin floor and elasticity |
| `optimize_promo_calendar` | Greedy ROI-ranked promo event scheduling within total budget + max-events-per-SKU constraints |
| `compute_competitive_price_index` | Volume-weighted Competitive Price Index (own price / competitor price × 100) by category / brand / pack size / market |

## Requirements

- Python 3.10+
- Dependencies: `mcp[cli]`, `pandas`, `pyarrow`, `numpy`, `scipy`

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

## Data format

The server expects NIQ-style flat files with these default column names (all overridable via tool parameters):

| Column | Default name | Description |
|---|---|---|
| Time period | `period_end_date` | Week-ending or month-ending date |
| SKU / UPC | `upc` | Product identifier |
| Market | `market` | Retail geography |
| Channel | `channel` | Trade channel (Grocery, Liquor, etc.) |
| Volume | `unit_sales` | Units sold |
| Dollar sales | `dollar_sales` | Revenue |
| Own price | `avg_price_per_unit` | Average shelf price |
| Promo flag | `any_promo_flag` | 1 = promoted week |
| TDP | `total_distribution_points` | Distribution measure |
| Trade spend | `trade_spend` | Promo cost (optional, for ROI) |
| Competitor brand | `competitor_brand` | Competitor identifier (pricing file) |
| Competitor price | `comp_avg_price` | Competitor average price (pricing file) |

All column names can be overridden when calling each tool.

## License

MIT
