# Feature Provenance & Temporal Safety System
## Comprehensive Planning Document

**Version:** 1.0  
**Status:** Planning Phase  
**Owner:** Processing Manager  
**Authority:** ServerBackend  

---

## Executive Summary

This document outlines the architecture for a comprehensive **Feature Provenance & Temporal Safety System** that ensures all features used in ML models can prove their calculation methodology, preventing future bias and ensuring model safety before production deployment.

### Core Principles

1. **Reproducibility**: Every feature must provide proof of how it was calculated
2. **Temporal Safety**: No feature can use future information (prevents look-ahead bias)
3. **Authority Model**: Processing Manager seeks approval from ServerBackend for registered features
4. **Production Safety Gate**: Models cannot deploy without passing feature provenance validation
5. **User Safety**: Prevents untrue/biased models from entering production pipeline

---

## System Architecture

### 1. Feature Registry (Central Authority)

**Location:** `Backend/app/core/processing/feature_registry.py`

```python
@dataclass
class FeatureProvenance:
    """Complete provenance record for a single feature."""
    
    # Identity
    feature_name: str
    feature_category: str  # 'technical', 'astronomical', 'snr', 'custom'
    version: str
    
    # Calculation Proof
    calculation_function: str  # Fully qualified function name
    source_file: str  # Path to source file
    source_line_number: int
    dependencies: List[str]  # Other features this depends on
    
    # Temporal Safety
    lookback_period: int  # How many bars back does it look?
    uses_future_data: bool  # CRITICAL: Must be False for production
    temporal_validation_method: str  # AST analysis, runtime check, manual review
    
    # Data Requirements
    required_columns: List[str]  # OHLCV columns needed
    optional_columns: List[str]
    minimum_data_points: int
    
    # Metadata
    description: str
    author: str
    created_date: datetime
    last_validated: datetime
    validation_status: str  # 'pending', 'approved', 'rejected'
    
    # Audit Trail
    audit_log: List[Dict[str, Any]]  # All changes/validations


class FeatureRegistry:
    """Central registry for all feature calculations."""
    
    def __init__(self, db_session):
        self.db = db_session
        self.registered_features: Dict[str, FeatureProvenance] = {}
        self.temporal_analyzer = TemporalAnalyzer()
    
    def register_feature(
        self,
        feature_name: str,
        calculation_function: Callable,
        category: str,
        lookback_period: int,
        required_columns: List[str],
        description: str,
        author: str = "system"
    ) -> FeatureProvenance:
        """
        Register a new feature with full provenance tracking.
        
        Steps:
        1. Extract function metadata (file, line number)
        2. Analyze for temporal safety (AST + runtime)
        3. Validate dependencies
        4. Create provenance record
        5. Store in database
        6. Request ServerBackend approval
        """
        pass
    
    def validate_feature_safety(self, feature_name: str) -> Dict[str, Any]:
        """
        Comprehensive safety validation for a feature.
        
        Returns:
            {
                'is_safe': bool,
                'temporal_violations': List[str],
                'dependency_issues': List[str],
                'recommendations': List[str]
            }
        """
        pass
    
    def get_feature_provenance(self, feature_name: str) -> Optional[FeatureProvenance]:
        """Retrieve complete provenance for a feature."""
        pass
    
    def request_serverbackend_approval(self, feature_name: str) -> bool:
        """
        Request approval from ServerBackend for feature usage.
        
        Processing Manager must seek authority before using features.
        """
        pass
```

---

### 2. Temporal Safety Analyzer

**Location:** `Backend/app/core/processing/temporal_analyzer.py`

```python
class TemporalAnalyzer:
    """Detects future bias in feature calculations."""
    
    def __init__(self):
        self.ast_analyzer = ASTFutureBiasDetector()
        self.runtime_monitor = RuntimeTemporalMonitor()
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """
        Multi-layer temporal safety analysis.
        
        Layers:
        1. AST Analysis: Static code inspection for future access patterns
        2. Runtime Monitoring: Track actual data access during execution
        3. Dependency Chain: Verify all dependencies are temporally safe
        
        Returns:
            {
                'uses_future_data': bool,
                'violations': List[str],
                'lookback_period': int,
                'confidence': float  # 0.0-1.0
            }
        """
        pass
    
    def detect_future_access_patterns(self, source_code: str) -> List[str]:
        """
        AST-based detection of future data access.
        
        Patterns to detect:
        - df.shift(-n) where n > 0
        - df.iloc[i+n] where n > 0 in loops
        - Rolling windows with forward-looking parameters
        - Direct index manipulation that accesses future rows
        """
        pass
    
    def monitor_runtime_access(self, func: Callable, test_data: pd.DataFrame) -> Dict:
        """
        Runtime monitoring of data access patterns.
        
        Wraps DataFrame operations to track:
        - Which rows are accessed
        - Temporal order of access
        - Any forward-looking operations
        """
        pass


class ASTFutureBiasDetector(ast.NodeVisitor):
    """AST visitor to detect future bias patterns."""
    
    def __init__(self):
        self.violations = []
        self.current_function = None
    
    def visit_Call(self, node):
        """Check for shift(-n), iloc[future], etc."""
        pass
    
    def visit_Subscript(self, node):
        """Check for df[i+n] patterns."""
        pass
```

---

### 3. Processing Manager Integration

**Location:** `Backend/app/core/processing/processing_manager.py` (modifications)

```python
class SafeProcessingManager(ProcessingManager):
    """Enhanced ProcessingManager with feature provenance enforcement."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_registry = FeatureRegistry(db_session)
        self.serverbackend_client = ServerBackendClient()
    
    async def execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute with feature provenance validation.
        
        New Steps:
        1. Identify all features that will be calculated
        2. Verify each feature is registered
        3. Request ServerBackend approval for feature set
        4. Validate temporal safety
        5. Execute with provenance tracking
        6. Log feature usage for audit
        """
        
        # Step 1: Identify features
        features_to_calculate = self._identify_required_features(self.analysis_type, self.config)
        
        # Step 2: Verify registration
        unregistered = []
        for feature in features_to_calculate:
            if not self.feature_registry.is_registered(feature):
                unregistered.append(feature)
        
        if unregistered:
            raise FeatureNotRegisteredException(
                f"Unregistered features detected: {unregistered}. "
                f"All features must be registered before use."
            )
        
        # Step 3: Request ServerBackend approval
        approval = await self.serverbackend_client.request_feature_approval(
            session_id=self.session_id,
            features=features_to_calculate,
            analysis_type=self.analysis_type
        )
        
        if not approval['approved']:
            raise FeatureApprovalDeniedException(
                f"ServerBackend denied feature approval: {approval['reason']}"
            )
        
        # Step 4: Validate temporal safety
        safety_report = self.feature_registry.validate_feature_set_safety(features_to_calculate)
        if not safety_report['is_safe']:
            raise TemporalSafetyViolationException(
                f"Temporal safety violations detected: {safety_report['violations']}"
            )
        
        # Step 5: Execute with provenance tracking
        result = await super().execute(df, **kwargs)
        
        # Step 6: Log usage
        await self._log_feature_usage(features_to_calculate, result)
        
        return result
```

---

### 4. Model Deployment Validator

**Location:** `Backend/app/core/ml/model_deployment_validator.py`

```python
class ModelDeploymentValidator:
    """Final safety gate before model deployment."""
    
    def __init__(self, feature_registry: FeatureRegistry):
        self.registry = feature_registry
    
    def validate_model_for_deployment(
        self,
        model_id: str,
        feature_list: List[str],
        training_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation before production deployment.
        
        Validation Checks:
        1. All features are registered
        2. All features have approved provenance
        3. No temporal safety violations
        4. All dependencies are satisfied
        5. Feature versions match training
        6. No deprecated features
        
        Returns:
            {
                'approved': bool,
                'validation_report': Dict,
                'blocking_issues': List[str],
                'warnings': List[str],
                'deployment_token': str  # If approved
            }
        """
        
        report = {
            'approved': False,
            'validation_report': {},
            'blocking_issues': [],
            'warnings': []
        }
        
        # Check 1: Registration
        for feature in feature_list:
            provenance = self.registry.get_feature_provenance(feature)
            if not provenance:
                report['blocking_issues'].append(
                    f"Feature '{feature}' is not registered"
                )
            elif provenance.uses_future_data:
                report['blocking_issues'].append(
                    f"Feature '{feature}' uses future data - TEMPORAL VIOLATION"
                )
            elif provenance.validation_status != 'approved':
                report['blocking_issues'].append(
                    f"Feature '{feature}' is not approved (status: {provenance.validation_status})"
                )
        
        # Check 2: Dependency validation
        dependency_issues = self._validate_dependency_chain(feature_list)
        report['blocking_issues'].extend(dependency_issues)
        
        # Check 3: Version compatibility
        version_warnings = self._check_version_compatibility(feature_list, training_metadata)
        report['warnings'].extend(version_warnings)
        
        # Final decision
        if not report['blocking_issues']:
            report['approved'] = True
            report['deployment_token'] = self._generate_deployment_token(model_id)
        
        return report
    
    def _generate_deployment_token(self, model_id: str) -> str:
        """Generate cryptographic token proving validation passed."""
        pass
```

---

### 5. ServerBackend Authority Model

**Location:** `Backend/app/api/serverbackend/feature_authority.py`

```python
class FeatureAuthorityService:
    """ServerBackend service for feature approval and authority."""
    
    def __init__(self, db_session):
        self.db = db_session
        self.approval_rules = self._load_approval_rules()
    
    async def approve_feature_request(
        self,
        session_id: str,
        features: List[str],
        analysis_type: str,
        requesting_manager: str
    ) -> Dict[str, Any]:
        """
        Process feature approval request from Processing Manager.
        
        Decision Factors:
        1. Feature registration status
        2. Temporal safety validation
        3. User permissions
        4. Resource availability
        5. Historical performance
        
        Returns:
            {
                'approved': bool,
                'reason': str,
                'approved_features': List[str],
                'denied_features': List[str],
                'conditions': List[str]  # Any conditions for approval
            }
        """
        pass
    
    def register_production_model(
        self,
        model_id: str,
        feature_list: List[str],
        validation_token: str
    ) -> Dict[str, Any]:
        """
        Register a validated model for production use.
        
        Models live on ServerBackend where users can download them.
        """
        pass
    
    def get_approved_features_for_user(self, user_id: str) -> List[str]:
        """Get list of features approved for a specific user."""
        pass
```

---

## Database Schema

### Feature Provenance Table

```sql
CREATE TABLE feature_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_name VARCHAR(255) UNIQUE NOT NULL,
    feature_category VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    
    -- Calculation Proof
    calculation_function TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_line_number INTEGER,
    dependencies JSONB DEFAULT '[]',
    
    -- Temporal Safety
    lookback_period INTEGER NOT NULL,
    uses_future_data BOOLEAN NOT NULL DEFAULT FALSE,
    temporal_validation_method VARCHAR(50),
    temporal_validation_confidence FLOAT,
    
    -- Data Requirements
    required_columns JSONB NOT NULL,
    optional_columns JSONB DEFAULT '[]',
    minimum_data_points INTEGER,
    
    -- Metadata
    description TEXT,
    author VARCHAR(100),
    created_date TIMESTAMP DEFAULT NOW(),
    last_validated TIMESTAMP,
    validation_status VARCHAR(20) DEFAULT 'pending',
    
    -- Audit
    audit_log JSONB DEFAULT '[]',
    
    -- Indexes
    INDEX idx_feature_name (feature_name),
    INDEX idx_category (feature_category),
    INDEX idx_validation_status (validation_status),
    INDEX idx_uses_future_data (uses_future_data)
);

CREATE TABLE feature_approval_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    requesting_manager VARCHAR(100),
    features JSONB NOT NULL,
    analysis_type VARCHAR(50),
    
    -- Decision
    approved BOOLEAN,
    decision_reason TEXT,
    decided_at TIMESTAMP,
    decided_by VARCHAR(100),
    
    -- Metadata
    requested_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_session_id (session_id),
    INDEX idx_approved (approved)
);

CREATE TABLE model_deployment_validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(255) UNIQUE NOT NULL,
    feature_list JSONB NOT NULL,
    
    -- Validation Results
    approved BOOLEAN NOT NULL,
    validation_report JSONB,
    blocking_issues JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    
    -- Deployment
    deployment_token TEXT,
    deployed_at TIMESTAMP,
    deployed_by VARCHAR(100),
    
    -- Metadata
    validated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_model_id (model_id),
    INDEX idx_approved (approved)
);
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

**Deliverables:**
- [ ] Create `FeatureProvenance` dataclass
- [ ] Implement `FeatureRegistry` core functionality
- [ ] Design database schema and migrations
- [ ] Create basic registration API

**Files to Create:**
- `Backend/app/core/processing/feature_registry.py`
- `Backend/app/core/processing/feature_provenance.py`
- `Backend/migrations/versions/XXXX_create_feature_provenance_tables.py`

### Phase 2: Temporal Safety (Weeks 3-4)

**Deliverables:**
- [ ] Implement `TemporalAnalyzer` with AST analysis
- [ ] Create runtime monitoring system
- [ ] Build test suite for temporal violations
- [ ] Document temporal safety patterns

**Files to Create:**
- `Backend/app/core/processing/temporal_analyzer.py`
- `Backend/app/core/processing/ast_future_bias_detector.py`
- `Backend/tests/test_temporal_safety.py`

### Phase 3: Integration & Authority (Weeks 5-6)

**Deliverables:**
- [ ] Integrate with ProcessingManager
- [ ] Implement ServerBackend authority service
- [ ] Create approval workflow
- [ ] Build admin UI for feature management

**Files to Modify:**
- `Backend/app/core/processing/processing_manager.py`
- `Backend/app/api/serverbackend/feature_authority.py`

**Files to Create:**
- `Backend/app/api/routes/features/feature_management.py`

### Phase 4: Deployment Safety Gate (Weeks 7-8)

**Deliverables:**
- [ ] Implement `ModelDeploymentValidator`
- [ ] Create deployment token system
- [ ] Build model registry on ServerBackend
- [ ] Create user download interface

**Files to Create:**
- `Backend/app/core/ml/model_deployment_validator.py`
- `Backend/app/api/routes/models/model_deployment.py`

---

## Feature Registration Process

### Step-by-Step Workflow

1. **Developer registers feature:**
   ```python
   registry.register_feature(
       feature_name="SMA_20",
       calculation_function=calculate_sma,
       category="technical",
       lookback_period=20,
       required_columns=["Close"],
       description="20-period Simple Moving Average"
   )
   ```

2. **System performs temporal analysis:**
   - AST inspection of `calculate_sma` function
   - Runtime test with sample data
   - Dependency chain validation

3. **Provenance record created:**
   - Function metadata extracted
   - Temporal safety score calculated
   - Database record inserted

4. **Approval workflow:**
   - Automatic approval if temporal safety score > 0.95
   - Manual review if 0.80 < score < 0.95
   - Automatic rejection if score < 0.80

5. **Feature becomes available:**
   - Processing Manager can request usage
   - ServerBackend grants approval
   - Feature used in analysis

---

## Temporal Safety Patterns

### ✅ SAFE Patterns

```python
# Lookback only
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# Shift backward
df['Prev_Close'] = df['Close'].shift(1)

# Cumulative operations
df['Cumulative_Volume'] = df['Volume'].cumsum()
```

### ❌ UNSAFE Patterns

```python
# Forward shift (FUTURE BIAS)
df['Next_Close'] = df['Close'].shift(-1)

# Forward-looking window
df['Future_High'] = df['High'].rolling(window=5).max().shift(-5)

# Direct future access
for i in range(len(df) - 5):
    df.loc[i, 'Future_Return'] = df.loc[i+5, 'Close'] / df.loc[i, 'Close']
```

---

## Next Steps

1. **Review this document** with the team
2. **Create FEATURE_REGISTRATION_INVENTORY.md** cataloging existing features
3. **Begin Phase 1 implementation** (Foundation)
4. **Set up development environment** for temporal analysis testing
5. **Design admin UI mockups** for feature management

---

## References

- Processing Manager: `Backend/app/core/processing/processing_manager.py`
- Technical Indicators: `Backend/app/core/analysis/technical_indicators.py`
- Astronomical Features: `Backend/app/core/analysis/astronomy/astronomical.py`
- SNR Signals: `Backend/app/core/analysis/trading/signal_generator.py`
- ML Pipeline: `Backend/app/core/ml/ml_dataset_preparation.py`

---

**Document Status:** ✅ Ready for Team Review  
**Last Updated:** 2026-05-19  
**Next Review:** After team feedback
