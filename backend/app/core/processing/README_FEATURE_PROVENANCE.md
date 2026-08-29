# Feature Provenance & Temporal Safety System
## Quick Start Guide

**Status:** Planning Complete ✅  
**Next Phase:** Implementation  
**Owner:** Processing Manager Team  

---

## 📋 Overview

This system ensures that every feature used in ML models can prove how it was calculated, preventing future bias and ensuring model safety before production deployment.

### Core Documents

1. **[FEATURE_PROVENANCE_SYSTEM.md](./FEATURE_PROVENANCE_SYSTEM.md)** - Complete architecture and implementation plan
2. **[FEATURE_REGISTRATION_INVENTORY.md](./FEATURE_REGISTRATION_INVENTORY.md)** - Catalog of all existing features to register

---

## 🎯 Key Objectives

1. **Reproducibility** - Every feature must provide proof of calculation methodology
2. **Temporal Safety** - No feature can use future information (prevents look-ahead bias)
3. **Authority Model** - Processing Manager seeks ServerBackend approval for features
4. **Production Safety Gate** - Models cannot deploy without passing provenance validation
5. **User Safety** - Prevents untrue/biased models from entering production

---

## 📊 Current Status

### Features Inventoried

| Module | Safe Features | Future-Looking Labels | Total |
|--------|---------------|----------------------|-------|
| Technical Indicators | ~120 | 0 | ~120 |
| Astronomical | ~80 | 0 | ~80 |
| SNR Signals | ~20 | ~30 | ~50 |
| **Total** | **~220** | **~30** | **~250** |

### Implementation Phases

| Phase | Timeline | Deliverables | Status |
|-------|----------|--------------|--------|
| **Phase 1: Foundation** | Weeks 1-2 | Feature Registry, Database Schema | 📋 Planned |
| **Phase 2: Temporal Safety** | Weeks 3-4 | AST Analyzer, Runtime Monitor | 📋 Planned |
| **Phase 3: Integration** | Weeks 5-6 | ProcessingManager Integration, ServerBackend Authority | 📋 Planned |
| **Phase 4: Deployment Gate** | Weeks 7-8 | Model Validator, Production Registry | 📋 Planned |

---

## 🏗️ System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ServerBackend (Authority)                 │
│  - Feature Approval Service                                  │
│  - Production Model Registry                                 │
│  - User Download Interface                                   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Approval Requests
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Processing Manager                         │
│  - Requests feature approval before use                      │
│  - Validates temporal safety                                 │
│  - Logs feature usage                                        │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Feature Lookup
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Feature Registry                          │
│  - Central feature catalog                                   │
│  - Provenance records                                        │
│  - Temporal safety validation                                │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Registration
                              │
┌─────────────────────────────────────────────────────────────┐
│              Feature Calculation Modules                     │
│  - Technical Indicators (120 features)                       │
│  - Astronomical Features (80 features)                       │
│  - SNR Signal Features (20 features)                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Registration Phase**
   - Developer registers feature with calculation function
   - System performs temporal analysis (AST + runtime)
   - Provenance record created in database
   - Approval workflow initiated

2. **Usage Phase**
   - Processing Manager identifies required features
   - Requests ServerBackend approval
   - Validates temporal safety
   - Executes with provenance tracking

3. **Deployment Phase**
   - Model Deployment Validator checks all features
   - Verifies no temporal violations
   - Generates deployment token
   - Model registered on ServerBackend

---

## 🔒 Temporal Safety

### ✅ Safe Patterns (Lookback Only)

```python
# Rolling windows
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# Backward shift
df['Prev_Close'] = df['Close'].shift(1)

# Cumulative operations
df['Cumulative_Volume'] = df['Volume'].cumsum()

# Historical analysis
df['Distance_To_Support'] = df['Close'] - df['Support_Level']
```

### ❌ Unsafe Patterns (Future Bias)

```python
# Forward shift - USES FUTURE DATA
df['Next_Close'] = df['Close'].shift(-1)

# Forward-looking window - USES FUTURE DATA
df['Future_High'] = df['High'].rolling(window=5).max().shift(-5)

# Direct future access - USES FUTURE DATA
for i in range(len(df) - 5):
    df.loc[i, 'Future_Return'] = df.loc[i+5, 'Close'] / df.loc[i, 'Close']
```

### Detection Methods

1. **AST Analysis** - Static code inspection for future access patterns
2. **Runtime Monitoring** - Track actual data access during execution
3. **Dependency Chain** - Verify all dependencies are temporally safe

---

## 📝 Registration Example

```python
from app.core.processing.feature_registry import FeatureRegistry
from app.core.analysis.technical_indicators import TechnicalIndicators

registry = FeatureRegistry(db_session)

# Register a simple moving average
registry.register_feature(
    feature_name="SMA_20",
    calculation_function=TechnicalIndicators._calculate_moving_averages,
    category="technical",
    lookback_period=20,
    required_columns=["Close"],
    description="20-period Simple Moving Average",
    author="system"
)

# System automatically:
# 1. Extracts function metadata (file, line number)
# 2. Analyzes for temporal safety (AST + runtime)
# 3. Validates dependencies
# 4. Creates provenance record
# 5. Stores in database
# 6. Requests ServerBackend approval
```

---

## 🚀 Quick Start for Developers

### 1. Review Planning Documents

- Read [FEATURE_PROVENANCE_SYSTEM.md](./FEATURE_PROVENANCE_SYSTEM.md) for complete architecture
- Review [FEATURE_REGISTRATION_INVENTORY.md](./FEATURE_REGISTRATION_INVENTORY.md) for existing features

### 2. Understand Your Module

Find your module in the inventory:
- **Technical Indicators** - ~120 features (SMA, EMA, RSI, MACD, etc.)
- **Astronomical** - ~80 features (planetary positions, aspects, moon phases)
- **SNR Signals** - ~20 safe features + ~30 future-looking labels

### 3. Identify Feature Priority

- **P0** (Critical) - Basic indicators, OHLCV features → Phase 1
- **P1** (Important) - Volume, crossovers, aspects → Phase 2
- **P2** (Advanced) - Patterns, SMC, asteroids → Phase 3

### 4. Prepare for Registration

For each feature in your module:
- [ ] Identify calculation function
- [ ] Calculate lookback period
- [ ] List required columns
- [ ] Verify temporal safety
- [ ] Write description
- [ ] Create test case

---

## ⚠️ Critical Notes

### Future-Looking Features

**~30 features in signal_generator.py use FUTURE DATA** - these are training labels, NOT features:

```python
# ❌ NEVER use as features for prediction
FUTURE_LOOKING_LABELS = [
    "max_favorable_move",
    "max_adverse_move", 
    "optimal_exit_price",
    "time_to_max_favorable",
    "level_respect_score",
    # ... etc
]

# ✅ Use ONLY as training targets
y_train = df["max_favorable_pct"]  # What we're trying to predict
X_train = df[SAFE_FEATURES]  # What we use to predict
```

These must be registered with `uses_future_data=True` flag for audit purposes.

### Processing Manager Authority

Processing Manager **MUST** seek ServerBackend approval before using any feature:

```python
# Before executing analysis
approval = await serverbackend.request_feature_approval(
    session_id=session_id,
    features=features_to_calculate,
    analysis_type=analysis_type
)

if not approval['approved']:
    raise FeatureApprovalDeniedException(approval['reason'])
```

### Model Deployment Gate

No model can deploy to production without passing validation:

```python
validator = ModelDeploymentValidator(feature_registry)

validation = validator.validate_model_for_deployment(
    model_id=model_id,
    feature_list=model_features,
    training_metadata=metadata
)

if not validation['approved']:
    raise DeploymentBlockedException(validation['blocking_issues'])
```

---

## 📚 Database Schema

### Core Tables

1. **feature_provenance** - Complete provenance records for all features
2. **feature_approval_requests** - Approval requests from Processing Manager
3. **model_deployment_validations** - Validation records for production models

See [FEATURE_PROVENANCE_SYSTEM.md](./FEATURE_PROVENANCE_SYSTEM.md) for complete schema.

---

## 🎯 Next Steps

### Immediate Actions (This Week)

1. **Team Review** - Review both planning documents with team
2. **Feedback Collection** - Gather input on architecture and approach
3. **Timeline Confirmation** - Confirm 8-week implementation timeline
4. **Resource Allocation** - Assign developers to each phase

### Phase 1 Kickoff (Next Week)

1. **Create Feature Registry** - Implement core FeatureRegistry class
2. **Design Database Schema** - Create migration for provenance tables
3. **Build Registration API** - Create endpoints for feature registration
4. **Start P0 Registration** - Begin registering critical features

---

## 📞 Contact & Support

- **Architecture Questions** - Review FEATURE_PROVENANCE_SYSTEM.md
- **Feature Inventory** - Review FEATURE_REGISTRATION_INVENTORY.md
- **Implementation Help** - Contact Processing Manager team

---

## 📖 Additional Resources

### Related Documentation

- Processing Manager: `Backend/app/core/processing/processing_manager.py`
- Technical Indicators: `Backend/app/core/analysis/technical_indicators.py`
- Astronomical Features: `Backend/app/core/analysis/astronomy/astronomical.py`
- SNR Signals: `Backend/app/core/analysis/trading/signal_generator.py`
- ML Pipeline: `Backend/app/core/ml/ml_dataset_preparation.py`

### External References

- [Preventing Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

**Document Version:** 1.0  
**Last Updated:** 2026-05-19  
**Status:** ✅ Ready for Team Review  
**Next Review:** After team feedback and Phase 1 kickoff
