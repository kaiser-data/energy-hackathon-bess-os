# Application Screenshots

This directory contains screenshots of the Smart Meter + BESS Analytics Platform for documentation purposes.

## Required Screenshots

### 1. Single Meter Dashboard (`single_meter_dashboard.png`)
- **Page**: Single Meter Analysis
- **Content**: Power analysis chart, daily energy consumption bars, power factor trends
- **Highlight**: Real-time monitoring capabilities and energy breakdown
- **Size**: 1920x1080 recommended

### 2. Compare Meters (`compare_meters.png`)
- **Page**: Compare Meters
- **Content**: Side-by-side meter comparison with overlay charts
- **Highlight**: Comparative analysis revealing consumption patterns
- **Size**: 1920x1080 recommended

### 3. BESS Overview (`bess_overview.png`)
- **Page**: BESS Overview Dashboard
- **Content**: SOC, SOH, thermal metrics, PCS data, alarm status
- **Highlight**: Comprehensive battery system monitoring
- **Size**: 1920x1080 recommended

### 4. PackPulse Overview (`packpulse_overview.png`)
- **Page**: Cell Analyzer - SAT Voltage Overview tab
- **Content**: System-wide voltage patterns with pack selection dropdown
- **Highlight**: Professional interface with time range controls
- **Size**: 1920x1080 recommended

### 5. PackPulse Heatmaps (`packpulse_heatmaps.png`)
- **Page**: Cell Analyzer - Heatmaps tab
- **Content**: Color-coded voltage distribution across 260 cells
- **Highlight**: Visual degradation patterns and cell health matrix
- **Size**: 1920x1080 recommended

### 6. PackPulse Trends (`packpulse_trends.png`)
- **Page**: Cell Analyzer - Pack Trends tab
- **Content**: Pack-level degradation analysis with trend lines
- **Highlight**: Statistical analysis and monthly degradation rates
- **Size**: 1920x1080 recommended

### 7. PackPulse Analysis (`packpulse_analysis.png`)
- **Page**: Cell Analyzer - Degradation Analysis tab
- **Content**: Curve fitting analysis with R² correlation metrics
- **Highlight**: Advanced analytics with quality assessment (German interface)
- **Size**: 1920x1080 recommended

## Screenshot Guidelines

### Technical Requirements
- **Format**: PNG with transparency where applicable
- **Resolution**: Minimum 1920x1080, preferably 2560x1440 for high-DPI displays
- **Browser**: Use Chrome/Firefox with full window capture
- **Data**: Use real BESS system data (ZHPESS232A230007 recommended)

### Content Guidelines
- **Time Range**: Use September 2024 - June 2025 for consistency
- **Data Quality**: Ensure all charts are fully loaded and rendered
- **UI State**: Show active/selected states where relevant
- **Professional Appearance**: Clean interface, no error messages visible

### Capture Process
1. Start both backend and frontend services
2. Navigate to each page/tab
3. Select appropriate data ranges and systems
4. Wait for full data loading (check loading indicators)
5. Capture full browser window or specific dashboard area
6. Verify image quality and content visibility

## File Naming Convention
- Use descriptive names matching the README references
- All lowercase with underscores for spaces
- Include page/feature identifier in filename
- Example: `packpulse_heatmaps.png`, `single_meter_dashboard.png`

## Alternative Text Descriptions
Each screenshot should include appropriate alt text in the README for accessibility and context when images cannot be displayed.