# Proyecto Final – Especialización Machine Learning Engineering · Curso IV

**Consultor MLOps:** Carlos Abanto  
**Fecha:** Febrero 2026  
**Cliente:** DSRPMart – Startup Marketplace  
**Casos de Uso Seleccionados:**  
1. Productos Recomendados (ranking por interacción, varias veces al día)  
2. Motor de Búsqueda (TOP-K productos por query del usuario)

**Proveedor Cloud:** Amazon Web Services (AWS) – Arquitectura Cloud-Native  
**Orquestación:** Apache Airflow (MWAA) + Kubeflow Pipelines (EKS)  
**Model Management:** MLflow (EKS)  
**Infraestructura:** Kubernetes (Amazon EKS)

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Matriz de Cumplimiento de Requerimientos](#2-matriz-de-cumplimiento-de-requerimientos)
3. [Nivel de Madurez MLOps](#3-nivel-de-madurez-mlops)
4. [Caso 1 – Productos Recomendados](#4-caso-1--productos-recomendados)
5. [Caso 2 – Motor de Búsqueda](#5-caso-2--motor-de-búsqueda)
6. [Tipo de Solución: Batch vs Realtime vs Streaming](#6-tipo-de-solución)
7. [Stack Tecnológico AWS Cloud-Native y Justificación Comparativa](#7-stack-tecnológico-aws-cloud-native-y-justificación-comparativa)
8. [Estrategia de Despliegue de Modelos](#8-estrategia-de-despliegue-de-modelos)
9. [Pasos de Construcción, Actores y Colaboración](#9-pasos-de-construcción-actores-y-colaboración)
10. [Diagramas de Arquitectura y Flujos de Proceso](#10-diagramas-de-arquitectura-y-flujos-de-proceso)
11. [Monitoreo, Data Drift y Observabilidad](#11-monitoreo-data-drift-y-observabilidad)
12. [Fuentes y Bibliografía](#12-fuentes-y-bibliografía)

---

## 1. Resumen Ejecutivo

DSRPMart es una startup de marketplace que necesita integrar predicciones de Machine Learning en múltiples componentes de su UI. Como consultores MLOps, proponemos una arquitectura **MLOps Nivel 2** (CT – Continuous Training) sobre **AWS con Kubernetes (EKS)** que garantiza:

- **Reproducibilidad** completa de experimentos y pipelines
- **Automatización** CI/CD desde el commit hasta producción
- **Escalabilidad** horizontal nativa con Kubernetes
- **Monitoreo** continuo de data drift, model performance y SLAs de negocio
- **Time-to-production** reducido para que los Data Scientists iteren rápidamente

Los dos casos seleccionados representan los extremos del espectro de frecuencia:
- **Productos Recomendados**: entrenamiento diario, inferencia batch cada 4-6 horas
- **Motor de Búsqueda**: indexación batch + retrieval en tiempo real (< 100ms)

Ambos comparten infraestructura, pipelines CI/CD y monitoring, maximizando la reutilización y reduciendo costos operativos.

---

## 2. Matriz de Cumplimiento de Requerimientos

La siguiente tabla mapea explícitamente cada requerimiento del proyecto con la sección del documento que lo desarrolla, asegurando **completitud y trazabilidad** para la evaluación.

| # | Requerimiento del Proyecto | Sección(es) que lo resuelven | Entregables concretos |
|---|---|---|---|
| **1** | Flujo E2E para cada caso de uso: algoritmos, fuentes de datos, optimizaciones, Model Card, Diccionario de Datos, Métricas de Negocio | §4 (Caso 1: §4.2–§4.6) y §5 (Caso 2: §5.2–§5.6) | Diagrama E2E, tabla de algoritmos con justificación, Model Card completa, catálogo de fuentes, KPIs de negocio |
| **2** | Tipo de Solución: batch, real-time o streaming con argumentación | §6 | Tabla comparativa por criterio, argumentación de descarte de alternativas |
| **3** | Stack Tecnológico: version control, cloud, IaC, model management, orquestación, librerías, CI/CD, métricas/monitoring, adicionales | §7 (§7.a–§7.h) + **§7.i Análisis Comparativo** | Tabla por categoría, config YAML, código Python, **matriz de decisión con alternativas evaluadas** |
| **4** | Estrategia de despliegue: shadow, backtest, champion-challenger, A/B test con diagrama de proceso | §8 | Diagrama de 5 fases, flujo Champion/Challenger → Shadow → A/B, criterios estadísticos |
| **5** | Pasos de construcción: desarrollos, actores/equipos, colaboración | §9 | Roadmap por Sprint (16 sem/caso), organigrama, modelo de colaboración Scrum adaptado |
| **6** | Diagramas de arquitectura: E2E training, arquitectura de solución, CI/CD de despliegue de modelo | §10 (§10.a–§10.c) | 3 diagramas de texto detallados: Pipeline E2E, Arquitectura AWS completa, CI/CD GitOps |

> **Nota sobre la evaluación:** La estructura del documento sigue el orden lógico de los 6 requerimientos. Cada caso de uso (§4 y §5) incluye su propio flujo E2E, algoritmos, Model Card, diccionario de datos y métricas de negocio como se solicita en el requerimiento 1. Los requerimientos 2–6 aplican de forma transversal a ambos casos.

---

## 3. Nivel de Madurez MLOps

Esta propuesta implementa **MLOps Nivel 2 (ML Pipeline Automation + CI/CD)** según la clasificación de Google Cloud / [Kreuzberger et al., 2023](https://arxiv.org/abs/2205.02302). A continuación se visualiza la progresión y se argumenta por qué este nivel es el apropiado para DSRPMart.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px', 'fontFamily': 'Segoe UI, Arial', 'primaryColor': '#f0f4f8', 'lineColor': '#4a5568'}}}%%
flowchart LR
    subgraph L0[" "]
        direction TB
        L0T["<b>NIVEL 0</b><br>Manual Process"]:::headerGray
        A0["❌ Training manual<br>❌ Deploy manual<br>❌ Sin CI/CD<br>❌ Sin monitoring"]:::gray
        L0T --- A0
    end

    subgraph L1[" "]
        direction TB
        L1T["<b>NIVEL 1</b><br>ML Pipeline Automation"]:::headerBlue
        A1["✅ Training automatizado<br>✅ Orquestación de pipelines<br>✅ Experiment Tracking<br>❌ Deploy manual"]:::blue
        L1T --- A1
    end

    subgraph L2[" "]
        direction TB
        L2T["<b>NIVEL 2 — DSRPMart</b><br>CI/CD + Continuous Training"]:::headerGreen
        A2["✅ CI/CD automatizado<br>✅ Continuous Training<br>✅ Model Registry<br>✅ Auto-monitoring & drift<br>✅ Feature Store"]:::green
        L2T --- A2
    end

    A0 ====>|"+ Airflow<br>+ Kubeflow Pipelines"| A1
    A1 ====>|"+ GitHub Actions + ArgoCD<br>+ Evidently AI + Feast"| A2

    classDef headerGray fill:#718096,stroke:#4a5568,color:#fff,font-weight:bold,stroke-width:2px
    classDef gray fill:#e2e8f0,stroke:#a0aec0,color:#2d3748,stroke-width:1px
    classDef headerBlue fill:#3182ce,stroke:#2c5282,color:#fff,font-weight:bold,stroke-width:2px
    classDef blue fill:#ebf8ff,stroke:#90cdf4,color:#2a4365,stroke-width:1px
    classDef headerGreen fill:#276749,stroke:#1a4731,color:#fff,font-weight:bold,stroke-width:2px
    classDef green fill:#c6f6d5,stroke:#68d391,color:#22543d,stroke-width:2px

    style L0 fill:#f7fafc,stroke:#cbd5e0,stroke-width:1px,rx:8
    style L1 fill:#f7fafc,stroke:#90cdf4,stroke-width:1px,rx:8
    style L2 fill:#f0fff4,stroke:#48bb78,stroke-width:3px,rx:8
```

**¿Por qué Nivel 2 y no Nivel 1?**

| Criterio | Nivel 1 (Pipeline) | Nivel 2 (CI/CD + CT) | Impacto para DSRPMart |
|---|---|---|---|
| **Deploy de modelo** | Manual (ML Engineer aprueba) | Automatizado con gates y rollback | Reduce time-to-production de días a horas |
| **Reentrenamiento** | Scheduled (pero manual trigger) | Continuous Training (drift-triggered) | Modelos siempre actualizados ante cambios de comportamiento |
| **Testing del pipeline** | Solo del modelo | Tests de código + datos + modelo + infra | Menor riesgo de errores en producción |
| **Reproducibilidad** | Parcial (solo training) | Completa (código + datos + config + infra) | Auditoría y compliance para inversores/reguladores |
| **Costo operativo** | Alto (intervención manual frecuente) | Bajo (automation > on-call) | Equipo de 11 personas puede operar 2+ modelos |

**¿Por qué NO Nivel 3 (Full Automation)?** No se justifica aún porque DSRPMart es una startup con 2 modelos en producción y 11 personas. El Nivel 3 con self-healing pipelines, automated model selection y zero-touch deployment agrega complejidad sin ROI proporcional a esta escala.

---

## 4. Caso 1 – Productos Recomendados

### 4.1 Descripción del Problema

Generar un **ranking personalizado de productos TOP-20** para cada usuario activo, actualizado varias veces al día basado en la interacción del usuario con la aplicación (clicks, vistas, compras, tiempo en página, búsquedas).

### 4.2 Flujo End-to-End

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   FLUJO E2E – PRODUCTOS RECOMENDADOS                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FUENTES DE DATOS                                                       │
│  ─────────────────                                                      │
│  [A] Amazon Kinesis Data Streams ← App Events (clicks, views, cart)     │
│      └── Kinesis Firehose → S3 (raw/events/) cada 5 min                │
│  [B] Amazon RDS (PostgreSQL) ← Catálogo de productos (CDC)             │
│      └── AWS DMS → S3 (raw/catalog/) CDC incremental                   │
│  [C] Amazon Redshift (DWH) ← Historial de compras/transacciones        │
│      └── UNLOAD → S3 (raw/transactions/) diario                        │
│  [D] Amazon ElastiCache Redis ← Sesión activa (features real-time)      │
│                                                                         │
│  PIPELINE DE ENTRENAMIENTO (Airflow MWAA → Kubeflow en EKS)            │
│  ──────────────────────────────────────────────────────────             │
│                                                                         │
│  [1] DATA INGESTION (Airflow Task)                                      │
│      ├── Lee: S3 raw/ (events, catalog, transactions)                   │
│      ├── Valida: Great Expectations (schema, nulls, ranges, freshness)  │
│      ├── Output: S3 validated/ (Parquet particionado por fecha)         │
│      └── Alerta: SNS → Slack si validación falla                       │
│           │                                                             │
│           ▼                                                             │
│  [2] FEATURE ENGINEERING (Kubeflow Component → Spark on EKS)            │
│      ├── User Features:                                                 │
│      │   ├── CTR por categoría (últimos 7/14/30 días)                   │
│      │   ├── Frecuencia de compra, ticket promedio, RFM score           │
│      │   ├── Session embedding (avg de item embeddings de sesión)       │
│      │   └── Hora del día, día de semana (features cíclicas sin/cos)   │
│      ├── Item Features:                                                 │
│      │   ├── Embedding Item2Vec (Word2Vec sobre secuencias de sesión)  │
│      │   ├── Popularidad (views, compras, add-to-cart últimas 24h)     │
│      │   ├── Precio, categoría L1/L2/L3, stock, margen                │
│      │   └── Freshness score (días desde publicación)                   │
│      ├── Cross Features:                                                │
│      │   ├── User×Category affinity matrix                              │
│      │   └── Cosine similarity (user_emb, item_emb)                    │
│      ├── Output: Feast Feature Store (offline → S3, online → Redis)     │
│      └── DVC: versiona dataset resultante en S3                        │
│           │                                                             │
│           ▼                                                             │
│  [3] MODEL TRAINING (Kubeflow Pipeline → Pod EKS con GPU p3.2xlarge)   │
│      ├── Stage A – Retrieval (Candidate Generation)                     │
│      │   └── Two-Tower Neural Network (TensorFlow)                      │
│      │       ├── User Tower: Dense(256)→Dense(128)→Dense(64)           │
│      │       ├── Item Tower: Dense(256)→Dense(128)→Dense(64)           │
│      │       ├── Loss: In-batch sampled softmax                         │
│      │       ├── Output: user_embedding + item_embedding (dim 64)       │
│      │       └── ANN Index: FAISS / Amazon OpenSearch KNN               │
│      ├── Stage B – Ranking                                              │
│      │   └── LightGBM LambdaRank                                       │
│      │       ├── Features: user+item+cross+retrieval_score              │
│      │       ├── Optimiza: NDCG@10 directamente                        │
│      │       └── Hyperparams: Optuna (Bayesian) 50 trials              │
│      ├── Stage C – Re-Ranking (diversidad)                              │
│      │   └── MMR (Maximal Marginal Relevance)                           │
│      │       └── λ=0.7 (relevance vs diversity trade-off)              │
│      ├── MLflow: log_params, log_metrics, log_artifact                  │
│      └── Tracking: experiment=product_recommender, run tagged con SHA   │
│           │                                                             │
│           ▼                                                             │
│  [4] MODEL EVALUATION (Kubeflow Component)                              │
│      ├── Métricas offline vs. Champion actual:                          │
│      │   ├── NDCG@10, NDCG@20                                          │
│      │   ├── Hit Rate@10, MAP@10                                        │
│      │   ├── MRR (Mean Reciprocal Rank)                                 │
│      │   └── Catalog Coverage (% de items únicos recomendados)          │
│      ├── Data Drift Check (Evidently AI):                               │
│      │   ├── PSI (Population Stability Index) por feature               │
│      │   ├── KS Test distribución de scores                             │
│      │   └── Target drift (distribución de clicks/compras)             │
│      ├── Decision Gate automático:                                      │
│      │   └── IF NDCG@10 > champion - 0.02 AND PSI < 0.25 → PASS       │
│      └── Output: Evidently HTML report → S3 + MLflow artifact          │
│           │                                                             │
│           ▼                                                             │
│  [5] MODEL REGISTRATION (Kubeflow Component)                            │
│      ├── MLflow Model Registry: create_model_version()                  │
│      ├── Stage: "Staging"                                               │
│      ├── Tags: git_sha, dataset_version (DVC), training_date            │
│      ├── Artifact: TF SavedModel + LightGBM .pkl → S3 artifact store   │
│      └── Notificación: SNS → Slack #ml-models                          │
│           │                                                             │
│           ▼                                                             │
│  [6] BATCH INFERENCE (Spark on EKS · Airflow scheduled 4x/día)         │
│      ├── Lee modelo desde MLflow (stage="Production")                   │
│      ├── Lee features desde Feast offline + online                      │
│      ├── Genera TOP-20 ranking por cada usuario activo                  │
│      ├── Output Redis (ElastiCache): key=user:{id}:recs TTL=6h         │
│      ├── Output S3 + Redshift: predictions/ (auditoría y analytics)     │
│      └── Métricas: cobertura, tiempo ejecución, p50/p99 latencia batch │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Algoritmos y Justificación

| Componente | Algoritmo | Por qué este algoritmo |
|---|---|---|
| **Retrieval** (Stage A) | **Two-Tower Neural Network** (TensorFlow) | Desacopla user/item towers permitiendo precomputar item embeddings offline. Escala a millones de productos con búsqueda ANN en < 10ms |
| **Ranking** (Stage B) | **LightGBM LambdaRank** | Optimiza NDCG directamente (learning-to-rank), rápido de entrenar, excelente con features tabulares + embeddings. Interpretable con SHAP |
| **Re-Ranking** (Stage C) | **MMR (Maximal Marginal Relevance)** | Evita "filter bubble" diversificando categorías en el top-20. Sin reentrenamiento, solo ajuste de λ |
| **Embeddings** | **Item2Vec** (Word2Vec sobre secuencias de sesión) | Captura co-ocurrencia de productos en sesiones reales (análogo a word embeddings) |
| **Cold Start** (usuarios nuevos) | **Content-Based + Popularidad** | Usuarios sin historial reciben top populares de su segmento demográfico |

**Optimizaciones aplicadas:**
- **Mixed Precision Training** (FP16) en Two-Tower → 2x más rápido en GPU
- **Feature hashing** para categorías de alta cardinalidad (> 10K SKUs)
- **Negative sampling** adaptativo (hard negatives de ANN) → mejor discriminación
- **Incremental training** diario (warm-start desde checkpoint anterior)

### 4.4 Model Card – Productos Recomendados

```
╔═══════════════════════════════════════════════════════════════════════════╗
║          MODEL CARD – DSRPMart Product Recommender v2.0                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  INFORMACIÓN GENERAL                                                      ║
║  ────────────────────                                                     ║
║  Nombre del modelo:    product_recommender                                ║
║  Versión:              2.0.0                                              ║
║  Tipo:                 Two-Tower NN (Retrieval) + LambdaRank (Ranking)    ║
║  Framework:            TensorFlow 2.15 + LightGBM 4.3                    ║
║  Propietario:          Equipo Data Science – DSRPMart                     ║
║  Fecha creación:       Febrero 2026                                       ║
║  Revisado por:         ML Lead / MLOps Lead                               ║
║  Frecuencia retrain:   Diario (incremental) + Semanal (full retrain)      ║
║                                                                           ║
║  DATOS DE ENTRENAMIENTO                                                   ║
║  ──────────────────────                                                   ║
║  Período:              Últimas 12 semanas (rolling window)                ║
║  Volumen:              ~150M eventos de interacción / ~10M usuarios        ║
║  Fuente principal:     S3 s3://dsrpmart-data/processed/events/            ║
║  Split estrategia:     Temporal – Train (semanas 1-9) / Val (10-11)       ║
║                        / Test (12). NO random split.                      ║
║  Preprocesamiento:     Spark on EKS → Feature Store Feast                 ║
║                                                                           ║
║  MÉTRICAS DE EVALUACIÓN (Offline – Test Set)                              ║
║  ────────────────────────────────────────────                             ║
║  ├── NDCG@10:               0.391                                         ║
║  ├── NDCG@20:               0.347                                         ║
║  ├── Hit Rate@10:           0.624                                         ║
║  ├── MAP@10:                0.218                                         ║
║  ├── MRR:                   0.302                                         ║
║  ├── Catalog Coverage:      71% (productos distintos en recs)             ║
║  └── Retrieval Recall@100:  0.87 (Two-Tower → top 100 candidates)        ║
║                                                                           ║
║  MÉTRICAS DE NEGOCIO IMPACTADAS                                           ║
║  ──────────────────────────────                                           ║
║  ├── CTR (Click-Through Rate) en sección "Para Ti"                        ║
║  ├── Add-to-Cart Rate desde recomendaciones                               ║
║  ├── Revenue per Session (uplift vs sin recomendaciones)                   ║
║  ├── Engagement: Tiempo promedio en app por sesión                        ║
║  └── Retention: D7 retention rate de usuarios activos                     ║
║                                                                           ║
║  LIMITACIONES Y SESGOS CONOCIDOS                                          ║
║  ────────────────────────────────                                         ║
║  - Usuarios con < 5 interacciones usan fallback de popularidad            ║
║  - Posible popularity bias: mitigado con MMR (diversity λ=0.7)            ║
║  - Rankings > 6h de antigüedad pueden no reflejar stock actualizado       ║
║  - No captura tendencias de minutos (ej: flash sale viral) sin streaming  ║
║                                                                           ║
║  USO PREVISTO                                                             ║
║  ─────────────                                                            ║
║  Generación batch de TOP-20 productos personalizados, actualizados 4      ║
║  veces al día (00:00, 06:00, 12:00, 18:00 UTC). Servido vía Redis         ║
║  ElastiCache con latencia < 5ms desde la API.                             ║
║                                                                           ║
║  UMBRALES DE ALERTA (Automated Guardrails)                                ║
║  ──────────────────────────────────────────                               ║
║  - NDCG@10 offline < 0.33       → bloquear despliegue                    ║
║  - CTR online < 0.07            → activar análisis de causa raíz          ║
║  - PSI cualquier feature > 0.25 → trigger reentrenamiento urgente         ║
║  - Coverage < 50%               → revisar pipeline de candidatos          ║
║  - Latencia Redis p99 > 10ms    → escalar ElastiCache                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 4.5 Diccionario / Catálogo de Fuentes de Datos

| # | Fuente | Sistema Origen | Destino / Ruta S3 | Columnas Clave | Formato | Frecuencia Actualización | Owner |
|---|---|---|---|---|---|---|---|
| 1 | Eventos de interacción (clicks, views, add-to-cart, purchase) | App Backend → Kinesis Data Streams | `s3://dsrpmart-data/raw/events/dt=YYYY-MM-DD/` | `user_id`, `product_id`, `event_type`, `timestamp`, `session_id`, `device`, `page` | Parquet (Firehose) | Near real-time (buffer 5 min) | Backend Team |
| 2 | Catálogo de productos | Amazon RDS PostgreSQL (CDC via DMS) | `s3://dsrpmart-data/raw/catalog/` | `product_id`, `title`, `category_l1`, `category_l2`, `category_l3`, `price`, `cost`, `stock`, `seller_id`, `created_at` | Parquet | CDC incremental (< 1 min) | Product Team |
| 3 | Historial de compras | Amazon Redshift DWH | `s3://dsrpmart-data/raw/transactions/dt=YYYY-MM-DD/` | `order_id`, `user_id`, `product_id`, `quantity`, `amount`, `discount_pct`, `payment_method`, `ts` | Parquet (UNLOAD) | Diario T+1h | Data Engineering |
| 4 | Perfil de usuario / segmentos | Amazon RDS PostgreSQL | `s3://dsrpmart-data/raw/users/` | `user_id`, `signup_date`, `age_range`, `city`, `segment`, `lifetime_value` | Parquet | Diario | CRM Team |
| 5 | Feature Store Online | Feast → Amazon ElastiCache Redis | N/A (in-memory) | `user_id` → `session_embedding`, `category_affinity`, `last_clicked_items` | Redis Hash | < 1 min (materialización Feast) | ML Platform |
| 6 | Feature Store Offline | Feast → S3 | `s3://dsrpmart-features/offline/` | `user_id` → `rfm_features`, `purchase_history_agg`, `ctr_by_category` | Parquet | 4x / día | ML Platform |
| 7 | Embeddings precomputados | Batch Job (Spark on EKS) | `s3://dsrpmart-models/embeddings/` | `item_id` → `embedding_64d`, `product_id` → `ann_index` | Parquet + FAISS index | Diario | Data Science |

### 4.6 Métricas de Negocio

#### Diagrama Mermaid – Ciclo de Vida del Modelo de Recomendaciones

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '13px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#4a5568'}}}%%
flowchart LR
    subgraph DATOS["DATA INGESTION"]
        direction TB
        K["<b>Kinesis</b><br>App Events"]:::aws
        RDS["<b>RDS</b><br>Catálogo"]:::aws
        RS["<b>Redshift</b><br>Transacciones"]:::aws
        K & RDS & RS --> S3R["<b>S3</b><br>raw/"]:::s3
    end

    subgraph FE["FEATURE ENGINEERING"]
        direction TB
        GE["<b>Great Expectations</b><br>Validación"]:::tool
        SPARK_FE["<b>Spark on EKS</b><br>Transformaciones"]:::tool
        FEAST["<b>Feast</b><br>Feature Store"]:::feast
        GE --> SPARK_FE --> FEAST
    end

    subgraph TRAIN["TRAINING ·  Kubeflow Pipeline"]
        direction TB
        TT["<b>Two-Tower NN</b><br>Retrieval · TensorFlow"]:::model
        LR2["<b>LambdaRank</b><br>Ranking · LightGBM"]:::model
        MMR["<b>MMR</b><br>Diversificación"]:::model
        TT --> LR2 --> MMR
    end

    subgraph EVAL["EVALUATION GATE"]
        direction TB
        EV{"NDCG@10 ≥ 0.33<br>PSI < 0.25"}:::decision
        REG["<b>MLflow Registry</b><br>stage = Staging"]:::mlflow
        ALERT["Alerta · Abort"]:::alert
        EV -->|"✅ PASS"| REG
        EV -->|"❌ FAIL"| ALERT
    end

    subgraph SERVE["SERVING LAYER"]
        direction TB
        SPARK_INF["<b>Spark Batch</b><br>Inference 4×/día"]:::tool
        REDIS["<b>ElastiCache</b><br>Redis · TTL 6h"]:::redis
        API["<b>FastAPI</b><br>GET /recs/:user_id"]:::api
        SPARK_INF --> REDIS --> API
    end

    S3R ==> GE
    FEAST ==> TT
    MMR ==> EV
    REG ==> SPARK_INF

    classDef aws fill:#232f3e,stroke:#ff9900,color:#ff9900,stroke-width:2px,font-weight:bold
    classDef s3 fill:#3f8624,stroke:#2d6a4f,color:#fff,stroke-width:2px
    classDef tool fill:#2b6cb0,stroke:#2c5282,color:#fff,stroke-width:1px
    classDef feast fill:#7b341e,stroke:#652b19,color:#fff,stroke-width:1px
    classDef model fill:#276749,stroke:#1a4731,color:#fff,stroke-width:2px
    classDef decision fill:#ecc94b,stroke:#d69e2e,color:#744210,stroke-width:2px
    classDef mlflow fill:#0077b6,stroke:#005f8a,color:#fff,stroke-width:2px
    classDef alert fill:#c53030,stroke:#9b2c2c,color:#fff,stroke-width:1px
    classDef redis fill:#dc2626,stroke:#b91c1c,color:#fff,stroke-width:1px
    classDef api fill:#6b21a8,stroke:#581c87,color:#fff,stroke-width:1px

    style DATOS fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,rx:10
    style FE fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px,rx:10
    style TRAIN fill:#dcfce7,stroke:#22c55e,stroke-width:2px,rx:10
    style EVAL fill:#fef9c3,stroke:#eab308,stroke-width:2px,rx:10
    style SERVE fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,rx:10
```

| KPI | Definición | Objetivo | Medición |
|---|---|---|---|
| **CTR@10** (Primario) | Clicks en top-10 recomendados / Impresiones top-10 | > 10% | Eventos Kinesis → Redshift dashboard |
| **Add-to-Cart Rate** | Add-to-cart desde recs / Impresiones | > 12% | Eventos app |
| **Revenue per Session** | Revenue atribuido a sesiones con recs / Total sesiones con recs | +8% vs. sin recs | A/B Test medición |
| **NDCG@10 offline** | Calidad del ranking en test set temporal | > 0.35 | MLflow automated eval |
| **Catalog Coverage** | Productos únicos recomendados / Total catálogo activo | > 60% | Batch job metric |
| **Latencia Serving** | p99 Redis GET user recommendations | < 5ms | CloudWatch + Prometheus |

---

## 5. Caso 2 – Motor de Búsqueda

### 5.1 Descripción del Problema

Conectar la **query/consulta del usuario** con los **TOP-K productos más relevantes** del catálogo, similar al proyecto DSRPflix. Requiere comprensión semántica de la query, matching con el inventario y ranking personalizado por usuario.

### 5.2 Flujo End-to-End

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  FLUJO E2E – MOTOR DE BÚSQUEDA                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FUENTES DE DATOS                                                       │
│  ─────────────────                                                      │
│  [A] Catálogo de productos (RDS → DMS → S3 CDC)                        │
│  [B] Historial de búsquedas + clicks (Kinesis → S3)                    │
│  [C] Queries de usuarios con resultados clickeados (ground truth)       │
│  [D] Catálogo de sinónimos / expansiones (curado por equipo de search)  │
│                                                                         │
│  PIPELINE OFFLINE (Indexación + Training)                                │
│  ────────────────────────────────────────                               │
│                                                                         │
│  [1] DATA PREPARATION (Airflow MWAA)                                    │
│      ├── Construir pares (query, product_clicked) del historial         │
│      ├── Negative sampling: productos mostrados pero NO clickeados      │
│      ├── Validar: Great Expectations (schema, volumen mínimo)           │
│      └── Output: S3 processed/search_training/                          │
│           │                                                             │
│           ▼                                                             │
│  [2] EMBEDDINGS GENERATION (Kubeflow → EKS GPU)                        │
│      ├── Modelo de embeddings: Sentence-BERT fine-tuned                 │
│      │   └── Base: all-MiniLM-L6-v2 (384 dims, rápido)                 │
│      │   └── Fine-tune con pares (query ↔ título+desc producto)         │
│      │   └── Contrastive Loss (positive pairs + hard negatives)        │
│      ├── Generar embeddings para TODOS los productos del catálogo       │
│      ├── Output: S3 embeddings/ + MLflow artifact                       │
│      └── MLflow: log model, params, eval metrics                        │
│           │                                                             │
│           ▼                                                             │
│  [3] INDEX CONSTRUCTION (Kubeflow Component)                            │
│      ├── Construir índice ANN (Approximate Nearest Neighbors)           │
│      │   └── Amazon OpenSearch KNN Plugin (HNSW algorithm)              │
│      │   └── Alternativa: FAISS IVF-PQ index en S3                     │
│      ├── Indexar catálogo completo con metadatos:                       │
│      │   └── product_id, title, category, price, stock, embedding       │
│      └── Output: OpenSearch índice "products-v{N}" (blue-green)         │
│           │                                                             │
│           ▼                                                             │
│  [4] RANKING MODEL TRAINING (Kubeflow → EKS)                           │
│      ├── Learning-to-Rank con LightGBM                                  │
│      │   ├── Features:                                                  │
│      │   │   ├── BM25 score (lexical match)                             │
│      │   │   ├── Semantic similarity (cosine query_emb × item_emb)      │
│      │   │   ├── Exact match score (query terms en título)              │
│      │   │   ├── Popularidad del producto (CTR, ventas)                 │
│      │   │   ├── User-item affinity (si user está logueado)             │
│      │   │   ├── Price competitiveness score                            │
│      │   │   └── Stock availability flag                                │
│      │   ├── Label: click (1), no-click (0), purchase (2)               │
│      │   └── Optimiza: NDCG@10                                         │
│      ├── MLflow: registro de experimento + modelo + métricas            │
│      └── Decision Gate → MLflow Model Registry (Staging)                │
│           │                                                             │
│           ▼                                                             │
│  [5] DAILY BATCH RE-INDEX (Airflow scheduled 02:00 UTC)                 │
│      ├── Recomputa embeddings de productos nuevos/modificados (CDC)     │
│      ├── Actualiza OpenSearch index (blue-green swap)                   │
│      ├── Actualiza modelo de ranking si nueva versión en MLflow         │
│      └── Warm-up cache de queries más frecuentes                       │
│                                                                         │
│  PIPELINE ONLINE (Serving en tiempo real)                                │
│  ─────────────────────────────────────────                              │
│                                                                         │
│  [6] QUERY-TIME FLOW (< 100ms end-to-end)                              │
│      ├── User query → API Gateway → Search Service (EKS Pod)           │
│      ├── Step 1: Query preprocessing                                    │
│      │   ├── Spell correction (SymSpell)                                │
│      │   ├── Query expansion (sinónimos)                                │
│      │   └── Tokenización + normalización                               │
│      ├── Step 2: Retrieval (paralelo)                                   │
│      │   ├── [Lexical] BM25 → OpenSearch text search → top 100         │
│      │   ├── [Semantic] Query embedding → OpenSearch KNN → top 100      │
│      │   └── Merge + dedup → ~150 candidates                           │
│      ├── Step 3: Ranking                                                │
│      │   ├── Feature computation on-the-fly (Redis features + basic)    │
│      │   ├── LightGBM predict → score 150 candidates                   │
│      │   └── Sort by score → top K (K configurable, default 20)         │
│      ├── Step 4: Post-processing                                        │
│      │   ├── Filtros: stock > 0, no productos reportados               │
│      │   ├── Business boost: productos sponsoreados (weighted)          │
│      │   └── Diversidad de sellers (max 3 resultados del mismo seller)  │
│      └── Response → API Gateway → Frontend (< 100ms p95)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Algoritmos y Justificación

| Componente | Algoritmo | Justificación |
|---|---|---|
| **Embeddings de búsqueda** | **Sentence-BERT** (all-MiniLM-L6-v2 fine-tuned) | Balance velocidad/calidad, 384 dims suficientes para e-commerce. Fine-tune en dominio de DSRPMart |
| **Retrieval lexical** | **BM25** (OpenSearch built-in) | Captura matches exactos que los embeddings pueden perder ("iPhone 15 Pro Max 256GB") |
| **Retrieval semántico** | **KNN HNSW** (Amazon OpenSearch) | Búsqueda ANN eficiente, managed service, < 20ms por query |
| **Ranking** | **LightGBM LambdaRank** | Combina signals lexicales + semánticas + negocio en un solo score optimizado para NDCG |
| **Spell correction** | **SymSpell** | O(1) lookup, pre-computable, ideal para latencia baja |
| **Query expansion** | **Sinónimos curados + Word2Vec** | Expande "celular" → "smartphone", "teléfono móvil" |

**Optimizaciones:**
- **Hybrid Search** (BM25 + KNN en paralelo) → mejor recall que cualquiera solo
- **Embedding quantization** (int8) → reduce tamaño del índice 4x, latencia 2x menor
- **Query result caching** en ElastiCache Redis (TTL 30 min) para queries frecuentes
- **Warm-up** de las top-1000 queries más populares al deploy del índice

### 5.4 Model Card – Motor de Búsqueda

```
╔═══════════════════════════════════════════════════════════════════════════╗
║          MODEL CARD – DSRPMart Search Engine v1.0                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  INFORMACIÓN GENERAL                                                      ║
║  ────────────────────                                                     ║
║  Nombre del modelo:    search_engine (embedding + ranker)                 ║
║  Versión:              1.0.0                                              ║
║  Tipo:                 Sentence-BERT (Retrieval) + LambdaRank (Ranking)   ║
║  Framework:            Sentence-Transformers 2.x + LightGBM 4.3          ║
║  Propietario:          Equipo Data Science – DSRPMart                     ║
║  Fecha creación:       Febrero 2026                                       ║
║  Frecuencia retrain:   Semanal (embeddings) + Diario (ranker)             ║
║                                                                           ║
║  DATOS DE ENTRENAMIENTO                                                   ║
║  ──────────────────────                                                   ║
║  Período:              Últimos 6 meses de búsquedas                        ║
║  Volumen:              ~50M pares (query, product_clicked)                 ║
║                        ~200K queries únicas / ~500K productos              ║
║  Fuente:               S3 s3://dsrpmart-data/processed/search/            ║
║  Split:                Temporal – Train 80% / Val 10% / Test 10%          ║
║                                                                           ║
║  MÉTRICAS DE EVALUACIÓN (Offline)                                         ║
║  ────────────────────────────────                                         ║
║  ├── NDCG@10:                0.452                                        ║
║  ├── MRR:                    0.387                                        ║
║  ├── Recall@100 (retrieval): 0.91                                         ║
║  ├── Precision@5:            0.34                                         ║
║  └── Zero-result rate:       < 2% de queries                              ║
║                                                                           ║
║  MÉTRICAS DE NEGOCIO IMPACTADAS                                           ║
║  ──────────────────────────────                                           ║
║  ├── Search CTR (clicks en resultados / búsquedas)                        ║
║  ├── Search Conversion Rate (compra tras búsqueda)                        ║
║  ├── Zero-result Rate (búsquedas sin resultados)                          ║
║  ├── Search Exit Rate (abandono tras búsqueda)                            ║
║  └── Revenue per Search (ingresos atribuidos a búsqueda)                  ║
║                                                                           ║
║  LIMITACIONES                                                             ║
║  ─────────────                                                            ║
║  - Nuevos productos (< 24h) solo tienen retrieval lexical hasta re-index  ║
║  - Queries muy largas (> 20 tokens) se truncan                            ║
║  - Idioma: solo español (Latam). No soporta queries en otros idiomas      ║
║  - Ranking personalizado solo para usuarios logueados con > 5 eventos     ║
║                                                                           ║
║  UMBRALES DE ALERTA                                                       ║
║  ────────────────                                                         ║
║  - Latencia p95 > 100ms     → escalar pods Search Service                ║
║  - Zero-result rate > 5%    → revisar índice y sinónimos                  ║
║  - Search CTR < 0.25        → análisis de relevancia + retraining         ║
║  - Embedding drift > 0.20   → re-fine-tune Sentence-BERT                 ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 5.5 Diccionario / Catálogo de Fuentes de Datos

| # | Fuente | Sistema | Ruta S3 / Endpoint | Columnas Clave | Frecuencia | Owner |
|---|---|---|---|---|---|---|
| 1 | Historial de búsquedas | Kinesis → S3 | `s3://dsrpmart-data/raw/search_logs/` | `query`, `user_id`, `results_shown[]`, `results_clicked[]`, `ts` | Near real-time | Backend |
| 2 | Catálogo de productos | RDS → DMS → S3 | `s3://dsrpmart-data/raw/catalog/` | `product_id`, `title`, `description`, `category_*`, `price`, `stock` | CDC < 1 min | Product |
| 3 | Sinónimos y expansiones | Repositorio Git (CSV) | `s3://dsrpmart-search/synonyms/` | `term`, `synonyms[]`, `category_scope` | Manual (PR) | Search Team |
| 4 | Product embeddings | Batch job diario | `s3://dsrpmart-models/search_embeddings/` | `product_id`, `embedding_384d` | Diario | Data Science |
| 5 | OpenSearch Index | Amazon OpenSearch Service | `https://search.dsrpmart.internal/products-v{N}` | Full product doc + embedding + metadata | Blue-green swap diario | ML Platform |
| 6 | Query cache | ElastiCache Redis | Redis cluster `search-cache` | `query_hash` → `[product_ids]` TTL 30 min | On query | Backend |

### 5.6 Métricas de Negocio

#### Diagrama Mermaid – Flujo Query-Time del Motor de Búsqueda (< 100ms)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '13px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#4a5568'}}}%%
flowchart LR
    USER["<b>Usuario</b><br>Ingresa query"]:::user

    subgraph PREPROCESS["QUERY PREPROCESSING  ~5ms"]
        QP["Spell Correction<br>Synonym Expansion<br>Tokenización"]:::process
    end

    subgraph RETRIEVAL["HYBRID RETRIEVAL  paralelo ~20ms"]
        BM25["<b>BM25 Lexical</b><br>OpenSearch<br>→ top 100"]:::lexical
        KNN["<b>KNN Semantic</b><br>HNSW 384d<br>→ top 100"]:::semantic
    end

    subgraph RANKING["RANKING  ~5ms"]
        MERGE["Merge + Dedup<br>~150 candidatos"]:::process
        RANK["<b>LightGBM</b><br>LambdaRank<br>150 → top 20"]:::ranker
        MERGE --> RANK
    end

    subgraph OUTPUT["POST-PROCESSING + RESPONSE"]
        POST["Stock filter<br>Seller diversity<br>Sponsored boost"]:::process
        CACHE["<b>Redis Cache</b><br>TTL=30min"]:::redis
        RESPONSE["<b>JSON Response</b><br>top-K productos<br><b>< 100ms p95</b>"]:::response
        POST --> CACHE --> RESPONSE
    end

    USER ==> QP
    QP ==> BM25 & KNN
    BM25 & KNN ==> MERGE
    RANK ==> POST

    classDef user fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px,font-weight:bold
    classDef process fill:#f8fafc,stroke:#94a3b8,color:#334155,stroke-width:1px
    classDef lexical fill:#1e40af,stroke:#1e3a8a,color:#fff,stroke-width:2px
    classDef semantic fill:#7c3aed,stroke:#6d28d9,color:#fff,stroke-width:2px
    classDef ranker fill:#c2410c,stroke:#9a3412,color:#fff,stroke-width:2px,font-weight:bold
    classDef redis fill:#dc2626,stroke:#b91c1c,color:#fff,stroke-width:1px
    classDef response fill:#047857,stroke:#065f46,color:#fff,stroke-width:3px,font-weight:bold

    style PREPROCESS fill:#f1f5f9,stroke:#94a3b8,stroke-width:1px,rx:8
    style RETRIEVAL fill:#eef2ff,stroke:#818cf8,stroke-width:2px,rx:8
    style RANKING fill:#fff7ed,stroke:#fb923c,stroke-width:2px,rx:8
    style OUTPUT fill:#ecfdf5,stroke:#34d399,stroke-width:2px,rx:8
```

| KPI | Definición | Objetivo | Medición |
|---|---|---|---|
| **Search CTR** (Primario) | Clicks en resultados / Total de búsquedas | > 30% | Kinesis → Redshift |
| **Zero-Result Rate** | Búsquedas sin ningún resultado / Total búsquedas | < 2% | Search Service logs |
| **Search Conversion** | Compras atribuidas a búsqueda / Total búsquedas con click | > 8% | Redshift attribution |
| **Latencia p95** | Tiempo de respuesta end-to-end de búsqueda | < 100ms | CloudWatch + Prometheus |
| **Revenue per Search** | Revenue atribuido a búsquedas / Total búsquedas | +10% vs. BM25 puro | A/B test |
| **Query Refinement Rate** | Usuarios que reformulan su query inmediatamente | < 15% | Session analysis |

---

## 6. Tipo de Solución

### 6.1 Productos Recomendados → **Batch (Mini-Batch Frecuente)**

| Criterio | Análisis | Decisión |
|---|---|---|
| Frecuencia requerida | "Varias veces al día" → No necesita real-time sub-segundo | **Batch cada 4-6h** |
| Volumen | ~10M usuarios × TOP-20 = 200M predicciones | Spark batch es más eficiente que real-time |
| Latencia aceptable | Rankings pre-calculados servidos desde Redis (< 5ms) | Batch + cache cumple SLA |
| Costo | Spot instances GPU para training, Spot CPU para inference → ~70% ahorro vs. on-demand | Batch optimiza costos |
| Complejidad del modelo | Two-Tower + LambdaRank son pesados para real-time per-request | Batch permite modelos más complejos |

**¿Por qué NO real-time?**
- Las recomendaciones no necesitan actualizarse en milisegundos; cada 4-6h es suficiente
- La personalización real-time requeriría servir el modelo Two-Tower + LambdaRank por request, con costos de GPU significativos
- El patrón batch → Redis da latencia < 5ms con costo muy inferior

**¿Por qué NO streaming?**
- No hay un evento individual que invalide todo el ranking (como un "stock agotado" que sí justificaría streaming)
- La complejidad de Spark Structured Streaming + Feature Store real-time no justifica el incremento marginal de frescura

### 6.2 Motor de Búsqueda → **Hybrid: Batch (Indexación) + Real-Time (Serving)**

| Criterio | Análisis | Decisión |
|---|---|---|
| Indexación de productos | Embeddings y ranking model se actualizan diario | **Batch** (Airflow scheduled) |
| Serving de queries | El usuario espera resultados inmediatos al escribir | **Real-time** (< 100ms p95) |
| SLA de latencia | UX de búsqueda requiere sub-segundo | Serving en EKS + Redis cache |
| Stock freshness | Productos sin stock no deben aparecer | CDC + filtro en query-time |

El Motor de Búsqueda es inherentemente **híbrido**: la indexación es batch pero el serving es real-time. La capa de ranking en query-time es un modelo ligero (LightGBM predict < 5ms para 150 candidatos) que se puede ejecutar por request sin problema.

---

## 7. Stack Tecnológico AWS Cloud-Native

### 7.a Control de Versiones de Código

| Herramienta | Uso | Justificación |
|---|---|---|
| **GitHub** (Organization) | Repositorios de código: modelos, pipelines, infraestructura | Estándar industria, code review, GitHub Actions |
| **DVC** (Data Version Control) | Versionado de datasets y modelos en S3 | Vincula versión de código con versión de datos. Reproducibilidad |
| **Git branching model** | `main` → prod, `develop` → staging, `feature/*` → desarrollo | Trunk-based para fast iteration |

**Estructura de repositorios:**
```
dsrpmart-org/
├── dsrpmart-ml-models/          # código de modelos (training, evaluation)
│   ├── product_recommender/
│   ├── search_engine/
│   ├── tests/
│   ├── dvc.yaml                 # pipelines DVC
│   └── .github/workflows/       # CI/CD
├── dsrpmart-ml-pipelines/       # Kubeflow Pipelines + Airflow DAGs
│   ├── kfp_components/
│   ├── airflow_dags/
│   └── .github/workflows/
├── dsrpmart-infra/              # Terraform + Helm Charts
│   ├── terraform/
│   │   ├── modules/
│   │   │   ├── eks/
│   │   │   ├── s3/
│   │   │   ├── redshift/
│   │   │   ├── elasticache/
│   │   │   ├── opensearch/
│   │   │   └── mwaa/
│   │   ├── environments/
│   │   │   ├── staging/
│   │   │   └── production/
│   │   └── backend.tf
│   └── helm/
│       ├── mlflow/
│       ├── kubeflow/
│       ├── feast/
│       └── monitoring/
└── dsrpmart-search-service/     # API de serving de búsqueda (FastAPI)
    ├── app/
    ├── Dockerfile
    └── .github/workflows/
```

### 7.b Proveedor de Nube, IaC y Administración del Sistema

| Componente | Herramienta AWS | Configuración |
|---|---|---|
| **Proveedor Cloud** | **Amazon Web Services (AWS)** | Región: us-east-1 (primary), us-west-2 (DR) |
| **IaC** | **Terraform** + **Terraform Cloud** | Módulos reutilizables por servicio, state remoto en S3 + DynamoDB lock |
| **Kubernetes** | **Amazon EKS** (Managed K8s) | v1.29, Managed Node Groups + Karpenter (auto-scaling) |
| **Helm Charts** | Helm 3 | Despliegue de MLflow, Kubeflow, Feast, Prometheus en EKS |
| **GitOps** | **ArgoCD** (en EKS) | Sincroniza estado del cluster con repositorio Git. Auto-heal |
| **Secrets** | **AWS Secrets Manager** | Rotación automática de credenciales DB, API keys, tokens |
| **IAM** | **IRSA** (IAM Roles for Service Accounts) | Pods K8s asumen roles IAM sin credentials hardcoded |
| **Networking** | VPC con subnets privadas + NAT Gateway | EKS en subnets privadas, Load Balancer en públicas |
| **DNS / Ingress** | **AWS ALB Ingress Controller** + Route53 | HTTPS en todos los endpoints, cert automático vía ACM |
| **Administración** | **AWS Systems Manager** + **AWS Config** | Compliance, patching, inventario de recursos |

**Arquitectura de Cuentas AWS (Multi-Account):**
```
AWS Organization
├── Management Account (billing, guardrails)
├── dsrpmart-dev       (desarrollo + experimentación)
├── dsrpmart-staging   (pre-producción, tests de integración)
└── dsrpmart-prod      (producción, datos reales)
```

### 7.c Herramienta de Model Management

| Herramienta | Función | Justificación |
|---|---|---|
| **MLflow Tracking Server** (desplegado en EKS) | Log de experimentos: params, métricas, artifacts | Open-source, agnóstico de framework, UI web |
| **MLflow Model Registry** | Versionado de modelos, transiciones Staging→Production | Centraliza lifecycle del modelo, API programable |
| **Backend Store** | Amazon RDS PostgreSQL (metadata de runs) | Managed, backups automáticos |
| **Artifact Store** | S3 `s3://dsrpmart-mlflow-artifacts/` | Almacena modelos serializados, FAISS indices, reports |

**Configuración MLflow:**
```yaml
# Helm values para MLflow en EKS
mlflow:
  tracking:
    backendStore: postgresql://mlflow:***@mlflow-db.internal:5432/mlflow
    artifactStore: s3://dsrpmart-mlflow-artifacts
    serviceAccount: mlflow-sa  # IRSA con acceso a S3
  resources:
    requests: { cpu: "1", memory: "2Gi" }
    limits:   { cpu: "2", memory: "4Gi" }
  ingress:
    host: mlflow.dsrpmart.internal
    auth: Cognito + ALB OIDC
```

### 7.d Orquestación y Entrenamiento de Modelos

| Herramienta | Rol | Justificación |
|---|---|---|
| **Apache Airflow** (Amazon MWAA) | Orquestación de workflows batch (scheduling, dependencias, retries) | Managed service, DAGs en Python, integraciones AWS nativas |
| **Kubeflow Pipelines** (KFP en EKS) | Pipelines de ML (training, evaluation, registration) | Componentes containerizados, trazabilidad, GPU scheduling |
| **Integración Airflow ↔ KFP** | Airflow triggerea KFP pipelines via `KubernetesPodOperator` o KFP SDK | Airflow como "meta-orquestador", KFP ejecuta ML-specific steps |
| **Katib** (Kubeflow) | Hyperparameter tuning (Bayesian, Grid, Random) | Parallelismo de trials en K8s, sin código custom |
| **Karpenter** + **Spot Instances** | Auto-scaling de nodos GPU/CPU para training | Reduce costos ~70% con Spot para workloads tolerantes |

**Ejemplo DAG Airflow – Productos Recomendados:**
```python
# dags/product_recommender_daily.py
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'on_failure_callback': slack_alert_callback,
}

with DAG(
    'product_recommender_daily',
    default_args=default_args,
    schedule_interval='0 0,6,12,18 * * *',  # 4 veces al día
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['ml', 'recommender', 'batch'],
) as dag:

    validate_data = KubernetesPodOperator(
        task_id='validate_data',
        image='dsrpmart/data-validator:latest',
        namespace='airflow',
        env_vars={'S3_PATH': 's3://dsrpmart-data/raw/events/'},
    )

    feature_engineering = KubernetesPodOperator(
        task_id='feature_engineering',
        image='dsrpmart/feature-eng:latest',
        namespace='kubeflow',
        container_resources={'requests': {'cpu': '4', 'memory': '16Gi'}},
    )

    trigger_kfp_training = KubernetesPodOperator(
        task_id='trigger_kfp_training',
        image='dsrpmart/kfp-trigger:latest',
        namespace='kubeflow',
        env_vars={'PIPELINE': 'product-recommender-train-v2'},
    )

    batch_inference = KubernetesPodOperator(
        task_id='batch_inference',
        image='dsrpmart/spark-inference:latest',
        namespace='spark',
        container_resources={'requests': {'cpu': '8', 'memory': '32Gi'}},
    )

    write_redis = KubernetesPodOperator(
        task_id='write_redis',
        image='dsrpmart/redis-writer:latest',
        namespace='serving',
    )

    validate_data >> feature_engineering >> trigger_kfp_training >> batch_inference >> write_redis
```

### 7.e Librerías, Frameworks y Herramientas

| Categoría | Herramientas | Uso |
|---|---|---|
| **Entrenamiento ML** | TensorFlow 2.15, LightGBM 4.3, Scikit-learn 1.4 | Two-Tower, LambdaRank, preprocessing |
| **NLP / Embeddings** | Sentence-Transformers, HuggingFace Transformers | Embeddings de búsqueda (Sentence-BERT) |
| **Procesamiento de datos** | Apache Spark 3.5 (on EKS via Spark Operator) | Feature engineering, batch inference a escala |
| **Feature Store** | Feast 0.38 (offline: S3/Parquet, online: Redis) | Features compartidas entre modelos, point-in-time correctness |
| **Serving API** | FastAPI + Uvicorn | API de búsqueda real-time, API de recomendaciones |
| **Búsqueda** | Amazon OpenSearch Service 2.x | Full-text search (BM25) + KNN vector search |
| **ANN Search** | FAISS (CPU) + OpenSearch KNN | Approximate nearest neighbors para embeddings |
| **Data Quality** | Great Expectations 0.18 | Validación de contratos de datos entre equipos |
| **Data Drift** | Evidently AI 0.4 | Detección de drift en features, target y predicciones |
| **Experiment Tracking** | MLflow 2.x | Logging de params, métricas, artifacts |
| **Contenedores** | Docker + Amazon ECR | Imágenes versionadas por commit SHA |
| **Cache / Serving Store** | Amazon ElastiCache (Redis 7.x) | Pre-computed recs + search cache |
| **DWH / Analytics** | Amazon Redshift Serverless | Queries analíticas, dashboards, A/B test analysis |
| **Streaming Ingesta** | Amazon Kinesis Data Streams + Firehose | Captura de eventos de app en near real-time |
| **CDC** | AWS DMS (Database Migration Service) | Replicación incremental de catálogo RDS → S3 |
| **Seguridad** | AWS IAM, IRSA, Secrets Manager, KMS, VPC, WAF | Zero-trust, encryption at rest y in transit |

### 7.f Solución para CI/CD

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CI/CD PIPELINE – GITHUB ACTIONS + ARGOCD          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  CI – GitHub Actions (on every PR + merge to main)           │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  Job 1: lint-and-test                                        │    │
│  │  ├── ruff lint (Python code quality)                         │    │
│  │  ├── mypy (static type checking)                             │    │
│  │  ├── pytest --cov (unit tests, >80% coverage)                │    │
│  │  ├── pytest tests/integration/ (integration tests en mock)   │    │
│  │  └── Great Expectations validation (sample data tests)       │    │
│  │                                                              │    │
│  │  Job 2: build-and-push (on merge to main)                    │    │
│  │  ├── docker build --platform linux/amd64                      │    │
│  │  ├── docker tag: {ECR_REPO}:{git-sha} + :latest              │    │
│  │  ├── docker push → Amazon ECR                                │    │
│  │  ├── trivy scan (container vulnerability scan)                │    │
│  │  └── Attest: cosign sign (supply chain security)             │    │
│  │                                                              │    │
│  │  Job 3: ml-pipeline-test (on merge to main)                  │    │
│  │  ├── DVC repro --dry (check reproducibility)                 │    │
│  │  ├── Trigger KFP pipeline on staging cluster                 │    │
│  │  ├── Wait for pipeline completion                            │    │
│  │  ├── Compare metrics vs. Champion (MLflow API)               │    │
│  │  └── Auto-comment PR with metric comparison table            │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│                      (on success)                                    │
│                           │                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  CD – ArgoCD GitOps (Continuous Deployment)                  │    │
│  ├──────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  Step 1: GitHub Actions updates Helm values                  │    │
│  │  ├── Updates image.tag in helm/*/values-prod.yaml            │    │
│  │  ├── Commits to dsrpmart-infra repo (gitops-prod branch)    │    │
│  │  └── Creates PR for approval (auto-merge for staging)        │    │
│  │                                                              │    │
│  │  Step 2: ArgoCD detects change in Git                        │    │
│  │  ├── Reconciles EKS cluster state with desired state         │    │
│  │  ├── Rolling update of pods (zero-downtime)                  │    │
│  │  └── Health checks (readinessProbe + livenessProbe)          │    │
│  │                                                              │    │
│  │  Step 3: Post-deploy validation                              │    │
│  │  ├── Smoke tests (automated e2e test in staging)             │    │
│  │  ├── MLflow model transition: Staging → Production           │    │
│  │  ├── Slack notification: #ml-deployments                     │    │
│  │  └── Rollback automático si health check falla               │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ROLLBACK PROCEDURE                                                  │
│  ─────────────────                                                   │
│  ├── Automático: ArgoCD detecta pod CrashLoop → reverts              │
│  ├── Manual: git revert en dsrpmart-infra → ArgoCD reconcilia       │
│  ├── MLflow: promote versión anterior → "Production"                 │
│  └── PagerDuty alert → on-call MLOps engineer                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.g Métricas de Performance, Aplicación y Herramientas de Visualización

| Tipo de Métrica | Métricas | Herramienta Recolección | Herramienta Visualización |
|---|---|---|---|
| **Model Performance (offline)** | NDCG@10, MRR, Hit Rate, Recall@100, RMSE | MLflow Tracking | MLflow UI + Grafana |
| **Model Performance (online)** | CTR, Conversion Rate, Revenue per Session | Kinesis → Redshift | Amazon QuickSight + Grafana |
| **Data Drift** | PSI por feature, KS test, target drift, embedding drift | Evidently AI (batch reports) | Evidently Dashboard + S3 HTML reports |
| **Infraestructura K8s** | CPU/Memory pods, pod restarts, node scaling | Prometheus (kube-state-metrics) | Grafana dashboards |
| **Aplicación / API** | Latencia p50/p95/p99, error rate, throughput (RPS) | Prometheus (FastAPI metrics) | Grafana |
| **Pipeline Health** | DAG success rate, task durations, SLA misses | Airflow metrics + CloudWatch | Grafana + Airflow UI |
| **Negocio** | GMV, Revenue Lift, Conversion, Retention | Redshift aggregations | Amazon QuickSight |
| **Costos** | Spend por servicio, costo por prediction, cost per experiment | AWS Cost Explorer + custom tags | QuickSight + Grafana |

**Stack de Observabilidad:**
```
Prometheus (EKS) ──────→ Grafana (dashboards)
    ├── kube-state-metrics        ├── Dashboard: ML Model Health
    ├── node-exporter             ├── Dashboard: Search Latency
    ├── custom app metrics        ├── Dashboard: Recommendation KPIs
    └── alertmanager ──→ PagerDuty (on-call)
                     ──→ Slack (#ml-alerts)

CloudWatch (AWS) ──────→ Grafana (vía CloudWatch datasource)
    ├── MWAA Airflow metrics
    ├── ElastiCache Redis metrics
    ├── OpenSearch metrics
    └── Kinesis throughput

Evidently AI (batch) ──→ S3 HTML reports + Grafana (custom panel)
    ├── Feature drift per model
    ├── Prediction drift
    └── Target drift
```

### 7.h Soluciones Adicionales

| Componente | Herramienta | Uso |
|---|---|---|
| **Feature Store** | **Feast** (offline: S3, online: ElastiCache Redis) | Features compartidas entre recomendador y buscador. Point-in-time joins correctos |
| **Vector Database** | **Amazon OpenSearch KNN** | Almacena y busca embeddings de productos para motor de búsqueda |
| **Data Catalog** | **AWS Glue Data Catalog** | Descubrimiento de datos, schemas, lineage |
| **Data Quality** | **Great Expectations** | Contratos de datos entre equipos (Data Engineering ↔ ML) |
| **Notebook Environment** | **JupyterHub en EKS** via Kubeflow Notebooks | Data Scientists pueden experimentar con acceso a GPU y datos |
| **Cost Management** | **Kubecost** (en EKS) | Atribución de costos por namespace/team/model |
| **Frontend** | **React / Next.js** | UI del marketplace consume APIs de recs y búsqueda |
| **API Gateway** | **Amazon API Gateway** + **ALB** | Rate limiting, auth, routing a microservicios en EKS |

### 7.i Análisis Comparativo del Stack – ¿Por Qué Estas Herramientas y No Otras?

> **Esta sección argumenta cada decisión tecnológica del stack comparando alternativas reales del mercado**, aplicando criterios objetivos de evaluación. Escala: ⭐ (1) a ⭐⭐⭐⭐⭐ (5).

#### 7.i.1 Proveedor de Nube: ¿Por qué AWS y no GCP o Azure?

| Criterio (peso) | AWS | GCP | Azure | Justificación de elección |
|---|:---:|:---:|:---:|---|
| **Servicios de ML managed** (25%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | SageMaker, Bedrock, OpenSearch nativo; Vertex AI de GCP es comparable pero AWS tiene mayor adopción enterprise |
| **Kubernetes managed (EKS)** (20%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GKE es ligeramente superior en UX, pero EKS + Karpenter cierra la brecha. AWS IRSA es mejor que GCP Workload Identity en flexibilidad |
| **Ecosistema de datos** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Kinesis + Redshift + S3 + Glue + DMS = ecosistema más maduro para data pipelines |
| **Talento disponible** (15%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | AWS tiene la mayor base de profesionales certificados; más fácil reclutar |
| **Costos Spot/Preemptible** (10%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | GCP Preemptible es más barato, pero AWS Spot tiene mejor disponibilidad y el ahorro total es similar |
| **Multi-region/DR** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | AWS tiene más regiones globales, importante para expansión de DSRPMart a Latam |
| **Score ponderado** | **4.55** | **4.20** | **3.80** | **→ AWS elegido** |

**Decisión:** AWS gana por ecosistema de datos maduro, talento disponible en Latam y mayor presencia regional. GCP sería segunda opción por su fortaleza en K8s y pricing.

#### 7.i.2 Orquestación: ¿Por qué Airflow (MWAA) + Kubeflow y no Prefect, Dagster o solo SageMaker Pipelines?

| Criterio (peso) | Airflow (MWAA) + KFP | Prefect 2.0 | Dagster | SageMaker Pipelines |
|---|:---:|:---:|:---:|:---:|
| **Madurez y comunidad** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **AWS managed service** (20%) | ⭐⭐⭐⭐⭐ (MWAA) | ⭐⭐ (self-hosted) | ⭐⭐ (self-hosted) | ⭐⭐⭐⭐⭐ |
| **ML-specific features** (20%) | ⭐⭐⭐⭐⭐ (KFP) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU scheduling en K8s** (15%) | ⭐⭐⭐⭐⭐ (KFP nativo) | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ (propio infra) |
| **Vendor lock-in** (15%) | ⭐⭐⭐⭐ (open-source) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ (SageMaker lock-in) |
| **DAGs como código Python** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Score ponderado** | **4.60** | **3.15** | **3.35** | **3.50** |

**Decisión:** La combinación **Airflow como meta-orquestador + Kubeflow Pipelines para ML** es únia en que: (1) MWAA elimina el ops de Airflow, (2) KFP da containerización nativa de cada step + GPU scheduling + experiment lineage, (3) ambos son open-source → sin vendor lock-in. SageMaker Pipelines se descartó por lock-in total al ecosistema propietario de AWS.

#### 7.i.3 Model Management: ¿Por qué MLflow y no Weights & Biases, Neptune AI o SageMaker Model Registry?

| Criterio (peso) | MLflow (self-hosted EKS) | W&B (SaaS) | Neptune AI (SaaS) | SageMaker Registry |
|---|:---:|:---:|:---:|:---:|
| **Costo** (25%) | ⭐⭐⭐⭐⭐ (gratis, OSS) | ⭐⭐ ($$$) | ⭐⭐⭐ ($$) | ⭐⭐⭐⭐ (incluido) |
| **Model Registry integrado** (20%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Framework agnóstico** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (bias a SM) |
| **API programable** (15%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Data residency control** (10%) | ⭐⭐⭐⭐⭐ (self-hosted) | ⭐⭐ (SaaS US) | ⭐⭐ (SaaS) | ⭐⭐⭐⭐ |
| **Comunidad y plugins** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Score ponderado** | **4.65** | **3.50** | **3.30** | **3.50** |

**Decisión:** MLflow self-hosted en EKS es la opción superior para una startup porque: (1) **costo $0** en licencias, solo infra (RDS + S3), (2) **control total** de datos y artifacts dentro de la VPC de AWS, (3) **API completa** para automatizar CI/CD pipelines, (4) **agnóstico** → funciona con TensorFlow, LightGBM, Sentence-Transformers sin adaptadores. W&B es excelente pero su costo SaaS (~$50/user/month) no se justifica en esta fase.

#### 7.i.4 Feature Store: ¿Por qué Feast y no Tecton o SageMaker Feature Store?

| Criterio | Feast (OSS) | Tecton | SageMaker Feature Store |
|---|:---:|:---:|:---:|
| **Costo** | ⭐⭐⭐⭐⭐ (gratis) | ⭐ ($$$$$) | ⭐⭐⭐ ($$) |
| **Point-in-time joins** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Online store: Redis** | ⭐⭐⭐⭐⭐ (nativo) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (DynamoDB) |
| **Offline store: S3/Parquet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Vendor lock-in** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Complejidad operacional** | ⭐⭐⭐ (self-managed) | ⭐⭐⭐⭐⭐ (SaaS) | ⭐⭐⭐⭐ |

**Decisión:** Feast es Open-Source, se despliega en el mismo EKS, usa ElastiCache Redis que ya tenemos para serving, y S3 para offline. No agrega costo adicional significativo. Tecton es superior en features pero su precio ($100K+/año) no escala para una startup.

#### 7.i.5 CI/CD: ¿Por qué GitHub Actions + ArgoCD y no Jenkins, GitLab CI o AWS CodePipeline?

| Criterio | GitHub Actions + ArgoCD | Jenkins | GitLab CI | AWS CodePipeline |
|---|:---:|:---:|:---:|:---:|
| **Integración con código** | ⭐⭐⭐⭐⭐ (GitHub nativo) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (GitLab) | ⭐⭐⭐ |
| **GitOps para K8s** | ⭐⭐⭐⭐⭐ (ArgoCD) | ⭐⭐ (plugin) | ⭐⭐⭐ | ⭐⭐ |
| **Costo operacional** | ⭐⭐⭐⭐ (SaaS + OSS) | ⭐⭐ (self-hosted) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **ML pipeline integration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Ecosistema marketplace** | ⭐⭐⭐⭐⭐ (Actions Market) | ⭐⭐⭐⭐ (plugins) | ⭐⭐⭐ | ⭐⭐ |
| **Auto-rollback K8s** | ⭐⭐⭐⭐⭐ (ArgoCD native) | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**Decisión:** GitHub Actions maneja CI (tests, build, push) y ArgoCD maneja CD (GitOps deployment a EKS). La separación CI ≠ CD sigue las mejores prácticas de GitOps. ArgoCD detecta drift en K8s y auto-heals, algo que CodePipeline o Jenkins no pueden hacer nativamente. Jenkins se descartó por alto costo operacional de mantener un Jenkins server.

#### 7.i.6 IaC: ¿Por qué Terraform y no AWS CDK, Pulumi o CloudFormation?

| Criterio | Terraform | AWS CDK | Pulumi | CloudFormation |
|---|:---:|:---:|:---:|:---:|
| **Multi-cloud portable** | ⭐⭐⭐⭐⭐ | ⭐ (AWS only) | ⭐⭐⭐⭐ | ⭐ (AWS only) |
| **Ecosistema providers** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **State management** | ⭐⭐⭐⭐ (S3 backend) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Comunidad y módulos** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Curva de aprendizaje** | ⭐⭐⭐⭐ (HCL) | ⭐⭐⭐⭐⭐ (TypeScript) | ⭐⭐⭐⭐ (Python) | ⭐⭐ (YAML/JSON) |
| **Plan/Preview** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (cdk diff) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (change set) |

**Decisión:** Terraform por su portabilidad (si DSRPMart migra a multi-cloud), madurez del ecosistema de providers, y la claridad del `terraform plan` que permite revisión en PR antes de aplicar cambios en infraestructura. AWS CDK sería segunda opción para equipos TypeScript-first.

#### 7.i.7 Mapa de Decisiones del Stack (Diagrama Mermaid)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD
    subgraph DECISIONS["MAPA DE DECISIONES TECNOLÓGICAS — DSRPMart"]
        direction TB

        subgraph ROW1[" "]
            direction LR
            Q1{"Cloud<br>Provider"}:::question -->|"Ecosistema datos · Talento Latam"| AWS["<b>AWS</b>"]:::answer
            Q2{"Orquesta-<br>ción"}:::question -->|"Managed + GPU K8s"| AK["<b>Airflow MWAA</b><br><b>+ Kubeflow KFP</b>"]:::answer
            Q3{"Model<br>Management"}:::question -->|"OSS · $0 · Agnóstico"| ML["<b>MLflow</b><br>self-hosted EKS"]:::answer
            Q4{"Feature<br>Store"}:::question -->|"OSS · Redis + S3"| FE2["<b>Feast</b>"]:::answer
        end

        subgraph ROW2[" "]
            direction LR
            Q5{"CI / CD"}:::question -->|"GitOps · Auto-heal K8s"| GA["<b>GitHub Actions</b><br><b>+ ArgoCD</b>"]:::answer
            Q6{"Infraestruc-<br>tura como<br>Código"}:::question -->|"Portable · Plan/Preview"| TF["<b>Terraform</b>"]:::answer
            Q7{"Serving<br>Layer"}:::question -->|"Batch→Redis · Realtime"| RF["<b>ElastiCache</b><br><b>+ FastAPI</b>"]:::answer
            Q8{"Vector<br>Search"}:::question -->|"Managed · BM25+KNN"| OS["<b>OpenSearch</b>"]:::answer
        end
    end

    AWS & AK & ML & FE2 & GA & TF & RF & OS -.->|"Desplegado en"| EKS["<b>Amazon EKS</b><br>Kubernetes v1.29 · Karpenter"]:::eks

    classDef question fill:#f8fafc,stroke:#94a3b8,color:#334155,stroke-width:2px
    classDef answer fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px,font-weight:bold
    classDef eks fill:#ff9900,stroke:#cc7a00,color:#232f3e,stroke-width:3px,font-weight:bold

    style DECISIONS fill:transparent,stroke:#e2e8f0,stroke-width:1px
    style ROW1 fill:#f1f5f9,stroke:#cbd5e0,stroke-width:1px,rx:8
    style ROW2 fill:#f1f5f9,stroke:#cbd5e0,stroke-width:1px,rx:8
```

---

## 8. Estrategia de Despliegue de Modelos

### 8.1 Estrategia: **Champion/Challenger + Shadow Mode → A/B Test**

#### Diagrama Mermaid – Flujo de Despliegue de Modelo

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '13px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD
    DEV["<b>FASE 0 · Desarrollo</b><br>JupyterHub · MLflow Experiments<br>Branch: feature/*"]:::phase0
    
    DEV -->|"PR + CI pass"| STAGING
    STAGING["<b>FASE 1 · Validación Staging</b><br>KFP Pipeline en staging EKS<br>Métricas offline vs Champion"]:::phase1
    
    STAGING -->|"NDCG pass + drift OK"| SHADOW
    SHADOW["<b>FASE 2 · Shadow Mode</b>  ·  1 semana<br>Challenger genera predicciones en paralelo<br>No se muestran al usuario"]:::phase2
    
    SHADOW -->|"Métricas estables"| AB
    AB["<b>FASE 3 · A/B Test</b>  ·  1-2 semanas<br>80% Champion  /  20% Challenger<br>KPIs: CTR · Conversion · Revenue"]:::phase3
    
    AB --> DECISION
    DECISION{"<b>¿Challenger supera Champion?</b><br>KPI lift > threshold · p < 0.05"}:::gate
    
    DECISION -->|"SÍ · Significativo"| PROMOTE
    DECISION -->|"NO"| ROLLBACK
    
    PROMOTE["<b>FASE 4a · PROMOCIÓN</b><br>MLflow: Challenger → Production<br>ArgoCD: full rollout 100%"]:::success
    ROLLBACK["<b>FASE 4b · ROLLBACK</b><br>Champion continúa operando<br>Post-mortem + learnings → backlog"]:::fail
    
    PROMOTE --> MONITOR["<b>Monitoring Intensivo 48h</b><br>Prometheus · Grafana · PagerDuty"]:::monitor
    ROLLBACK --> BACKLOG["Nuevos experimentos"]:::neutral

    classDef phase0 fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px
    classDef phase1 fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:2px
    classDef phase2 fill:#7c3aed,stroke:#6d28d9,color:#fff,stroke-width:2px
    classDef phase3 fill:#ca8a04,stroke:#a16207,color:#fff,stroke-width:2px
    classDef gate fill:#fef3c7,stroke:#d97706,color:#92400e,stroke-width:3px,font-weight:bold
    classDef success fill:#15803d,stroke:#166534,color:#fff,stroke-width:3px,font-weight:bold
    classDef fail fill:#b91c1c,stroke:#991b1b,color:#fff,stroke-width:2px
    classDef monitor fill:#0e7490,stroke:#155e75,color:#fff,stroke-width:1px
    classDef neutral fill:#f1f5f9,stroke:#94a3b8,color:#475569,stroke-width:1px
```

Se utiliza la **misma estrategia para ambos modelos** porque:
1. Ambos tienen componente batch (se puede hacer shadow sin costo de latencia)
2. Ambos impactan métricas de negocio medibles (CTR, conversion, revenue)
3. La infraestructura compartida (EKS + Redis + MLflow) soporta multi-version nativa
4. Los A/B tests necesitan significancia estadística → requiere volumen de tráfico que DSRPMart tiene

### 8.2 Diagrama de Proceso de Despliegue

```
╔══════════════════════════════════════════════════════════════════════════╗
║       PROCESO DE DESPLIEGUE DE MODELOS – DSRPMart (ambos modelos)       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  FASE 0: DESARROLLO + EXPERIMENTACIÓN                                   ║
║  ────────────────────────────────────                                    ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │  Data Scientist trabaja en branch feature/                     │      ║
║  │  ├── Experimenta en JupyterHub (Kubeflow Notebooks)           │      ║
║  │  ├── Registra experimentos en MLflow (experiment tracking)    │      ║
║  │  ├── Usa Feast para acceder a features (offline store)         │      ║
║  │  ├── DVC tracks dataset version                               │      ║
║  │  └── Cuando métricas offline son prometedoras → PR a develop  │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║  FASE 1: VALIDACIÓN EN STAGING                                           ║
║  ─────────────────────────────                                           ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │  PR merged → GitHub Actions CI/CD                              │      ║
║  │  ├── Tests pasan (unit + integration + data quality)           │      ║
║  │  ├── Docker image built → pushed a ECR (:sha)                  │      ║
║  │  ├── KFP Pipeline triggered en staging EKS                     │      ║
║  │  │   ├── Train con datos staging (subset de prod data)        │      ║
║  │  │   ├── Evaluate: métricas offline vs. Champion actual       │      ║
║  │  │   └── Evidently report: drift check                        │      ║
║  │  ├── IF pass → MLflow register model (stage=Staging)           │      ║
║  │  └── Slack notification: "Model v{X} ready for shadow"        │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║  FASE 2: SHADOW MODE (1 semana)                                         ║
║  ──────────────────────────────                                          ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │  Challenger genera predicciones en PARALELO al Champion        │      ║
║  │  ├── Recomendador: batch inference → Redis ns "challenger"     │      ║
║  │  │   (NO se muestran al usuario, solo se almacenan)            │      ║
║  │  ├── Buscador: ranking challenger vs champion (logged only)    │      ║
║  │  │   (usuario ve resultados del Champion, Challenger logs)     │      ║
║  │  ├── Comparación automática offline:                           │      ║
║  │  │   ├── NDCG, MRR, Coverage (diario)                         │      ║
║  │  │   └── Evidently drift report                                │      ║
║  │  └── Dashboard Grafana: Champion vs. Challenger side-by-side  │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║       ▼                                                                  ║
║  FASE 3: A/B TEST (1-2 semanas)                                         ║
║  ───────────────────────────────                                         ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │  Split de tráfico: 80% Champion / 20% Challenger               │      ║
║  │  ├── Recomendador:                                             │      ║
║  │  │   ├── Redis key: user:{id}:recs (champion group)            │      ║
║  │  │   ├── Redis key: user:{id}:recs:challenger (test group)     │      ║
║  │  │   └── Frontend: routing por user_id % 100 (determinista)    │      ║
║  │  ├── Buscador:                                                 │      ║
║  │  │   ├── Search Service: model_version param en request        │      ║
║  │  │   └── Routing por user_id hash (sticky sessions)            │      ║
║  │  ├── Medición de KPIs por grupo:                               │      ║
║  │  │   ├── CTR, Conversion, Revenue, Session Time                │      ║
║  │  │   ├── Statistical significance: t-test (p < 0.05)          │      ║
║  │  │   └── Minimum Detectable Effect: 2% CTR lift               │      ║
║  │  └── Dashboard: Redshift → QuickSight A/B analysis            │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║       │                                                                  ║
║     ┌─┴──────────────────────────────────┐                              ║
║     │  ¿Challenger supera Champion?       │                              ║
║     │  (KPI lift > threshold + p<0.05)    │                              ║
║     └─┬────────────────────────────┬──────┘                              ║
║       │ SÍ                         │ NO                                  ║
║       ▼                            ▼                                     ║
║  FASE 4a: PROMOCIÓN            FASE 4b: ROLLBACK                        ║
║  ─────────────────             ──────────────────                        ║
║  ┌──────────────────────┐     ┌──────────────────────┐                  ║
║  │ MLflow: Challenger    │     │ Challenger archivado  │                  ║
║  │ → stage="Production"  │     │ Champion continúa     │                  ║
║  │ Champion              │     │ Post-mortem doc       │                  ║
║  │ → stage="Archived"    │     │ Learnings → backlog   │                  ║
║  │                       │     └──────────────────────┘                  ║
║  │ ArgoCD: update        │                                               ║
║  │ ConfigMap model_ver   │                                               ║
║  │                       │                                               ║
║  │ Full rollout 100%     │                                               ║
║  │ Monitoring 48h        │                                               ║
║  └──────────────────────┘                                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 9. Pasos de Construcción, Actores y Colaboración

### 9.1 Plan de Construcción – Caso 1: Productos Recomendados

| Fase | Sprint | Tareas | Duración | Responsable |
|---|---|---|---|---|
| **Discovery** | Sprint 0 | Levantamiento de requerimientos con Product, definir KPIs/SLAs, audit de fuentes de datos | 1 semana | ML Lead + Product Owner |
| **Infraestructura** | Sprint 1 | Terraform: EKS, S3, ElastiCache, MWAA. Helm: MLflow, Kubeflow, Feast, Prometheus | 2 semanas | ML Platform + DevOps |
| **Data Pipeline** | Sprint 2 | Kinesis ingesta, DMS CDC, Airflow DAG ingestion, Great Expectations validación | 2 semanas | Data Engineering |
| **Features** | Sprint 3 | Spark feature eng, Item2Vec embeddings, Feast offline+online store | 2 semanas | Data Scientist + ML Platform |
| **Modelo v1** | Sprint 4 | Two-Tower training, LambdaRank ranking, evaluation pipeline, MLflow registro | 2 semanas | Data Scientist Senior |
| **Serving** | Sprint 5 | Spark batch inference, Redis writer, FastAPI recomendaciones, integración frontend | 2 semanas | MLOps + Backend |
| **CI/CD** | Sprint 5 | GitHub Actions workflows, ArgoCD setup, ECR pipelines | 1 semana | MLOps / DevOps |
| **Monitoring** | Sprint 6 | Evidently drift, Prometheus alertas, Grafana dashboards, PagerDuty integración | 1 semana | MLOps |
| **Rollout** | Sprint 7-8 | Shadow mode (1 sem) + A/B test (2 sem) + promoción | 3 semanas | ML Lead + QA + Product |
| **Total** | | | **~16 semanas** | |

### 9.2 Plan de Construcción – Caso 2: Motor de Búsqueda

| Fase | Sprint | Tareas | Duración | Responsable |
|---|---|---|---|---|
| **Discovery** | Sprint 0 | Análisis de queries de usuario, gap analysis, definir KPIs búsqueda | 1 semana | ML Lead + Product |
| **OpenSearch Setup** | Sprint 1 | Terraform OpenSearch, índice base BM25, integración con catálogo CDC | 2 semanas | ML Platform + DE |
| **Embeddings** | Sprint 2 | Fine-tune Sentence-BERT, pipeline de embedding generation, KNN index | 2 semanas | Data Scientist |
| **Hybrid Retrieval** | Sprint 3 | BM25 + KNN paralelo, merge/dedup, SymSpell + sinónimos | 2 semanas | Data Scientist + Backend |
| **Ranking Model** | Sprint 4 | LightGBM LambdaRank, feature engineering para search, MLflow tracking | 2 semanas | Data Scientist |
| **Search Service** | Sprint 5 | FastAPI serving, Redis cache, query preprocessing pipeline | 2 semanas | Backend + MLOps |
| **CI/CD + Blue-Green** | Sprint 5 | Index versioning OpenSearch, GitHub Actions, ArgoCD | 1 semana | MLOps / DevOps |
| **Monitoring** | Sprint 6 | Search metrics (latency, zero-result, CTR), Evidently drift, Grafana | 1 semana | MLOps |
| **Rollout** | Sprint 7-8 | Shadow mode + A/B test vs. BM25 baseline | 3 semanas | ML Lead + Product |
| **Total** | | | **~16 semanas** | |

### 9.3 Actores y Equipos

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    ORGANIZACIÓN DEL EQUIPO DSRPMart ML                   ║
╠════════════════════════╦═════════════════════════════════════════════════╣
║  Rol                   ║  Responsabilidades                              ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  ML Lead / Arquitecto  ║  Diseño de arquitectura MLOps, Model Cards,     ║
║  (1 persona)           ║  decisiones de stack, go/no-go de despliegues,  ║
║                        ║  revisión técnica de modelos y pipelines         ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  Data Scientist Sr     ║  Two-Tower, Sentence-BERT: modelos complejos,   ║
║  (1 persona)           ║  research de arquitecturas, tuning avanzado     ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  Data Scientist        ║  LambdaRank, Item2Vec, feature engineering,     ║
║  (2 personas)          ║  EDA, experimentación, evaluación offline       ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  MLOps Engineer        ║  Kubeflow Pipelines, Airflow DAGs, CI/CD,       ║
║  (2 personas)          ║  batch inference, monitoring, Evidently,        ║
║                        ║  MLflow administration, on-call rotación        ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  Data Engineer         ║  Kinesis ingesta, Spark jobs, DMS CDC,          ║
║  (1 persona)           ║  Redshift DWH, data quality contracts           ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  ML Platform Engineer  ║  EKS cluster, Terraform, ArgoCD, Feast,        ║
║  (1 persona)           ║  OpenSearch, ElastiCache, seguridad (IRSA)      ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  Backend Engineer      ║  FastAPI Search Service, API Gateway, frontend  ║
║  (1 persona)           ║  integration, Redis client optimization         ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  Product Owner         ║  Definir KPIs, priorizar features, stakeholder  ║
║  (1 persona)           ║  management, A/B test analysis review           ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  QA / Analytics        ║  A/B test statistical analysis, data validation,║
║  (1 persona)           ║  regression testing, UAT                        ║
╠════════════════════════╬═════════════════════════════════════════════════╣
║  TOTAL: 11 personas                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 9.4 Modelo de Colaboración

```
PROCESO DE TRABAJO – ML-ADAPTED SCRUM
──────────────────────────────────────

CADENCIA
├── Sprint: 2 semanas
├── Daily Standup: 15 min (async en Slack los viernes)
├── Sprint Planning: lunes S1 (2h)
├── Sprint Review: viernes S2 (1h) – demo de métricas + pipeline
├── Retro: viernes S2 (30 min)
└── ML Review: miércoles S2 (1h) – revisión técnica de modelos

ARTEFACTOS
├── Product Backlog: Jira Board "DSRPMart ML"
├── Sprint Backlog: Jira + GitHub Projects
├── RFC (Request For Comments): Documento técnico en Notion
│   └── Requerido para: nuevo modelo, cambio de stack, nuevo pipeline
├── Model Card: actualizado en cada release (en repo Git)
├── Runbook: procedimientos de on-call y rollback en Confluence
└── ADR (Architecture Decision Records): en repo dsrpmart-infra/

COLABORACIÓN ENTRE EQUIPOS
├── Data Scientist ↔ MLOps:
│   ├── DS entrega: notebook reproducible + MLflow experiment ID
│   └── MLOps convierte a: KFP Pipeline + Airflow DAG + CI/CD
│
├── Data Engineering ↔ ML:
│   ├── Contrato de datos: Great Expectations suite compartida
│   ├── DE publica features en Feast offline store
│   └── ML consume vía Feast SDK (100% reproducible)
│
├── Backend ↔ ML:
│   ├── API contract: OpenAPI spec (search, recommendations)
│   ├── Redis key naming convention documentada
│   └── SLA: latencia p99, throughput, error budget
│
└── Product ↔ ML:
    ├── KPI definition doc (antes de cada modelo)
    ├── A/B test plan con sample size calculado
    └── Go/No-Go decision meeting post A/B test
```

---

## 10. Diagramas de Arquitectura y Flujos de Proceso

### 10.a End-to-End Entrenamiento de Modelo (Ambos Modelos)

```
╔══════════════════════════════════════════════════════════════════════════╗
║    PIPELINE E2E ENTRENAMIENTO – AWS + Airflow + Kubeflow + MLflow        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────┐                    ║
║  │  AMAZON MWAA (Managed Airflow)                    │                    ║
║  │  DAG: {model}_training_pipeline                   │                    ║
║  │  Schedule:                                        │                    ║
║  │    - product_recommender: daily 00:00 UTC         │                    ║
║  │    - search_engine:       daily 02:00 UTC         │                    ║
║  └───────────────────────┬─────────────────────────┘                    ║
║                          │                                               ║
║      ┌───────────────────┼───────────────────────┐                      ║
║      ▼                   ▼                       ▼                      ║
║  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐              ║
║  │ Task 1:  │    │ Task 2:      │    │ Task 3:          │              ║
║  │ S3 Sensor│───→│ Data Validate│───→│ Feature Eng      │              ║
║  │(new data)│    │ Great Expect.│    │ Spark on EKS     │              ║
║  │          │    │ Schema+Nulls │    │ → Feast Store    │              ║
║  │ S3 raw/  │    │ → S3 valid/  │    │ → S3 features/   │              ║
║  └──────────┘    └──────────────┘    └────────┬─────────┘              ║
║                                               │                        ║
║                                               ▼                        ║
║  ┌────────────────────────────────────────────────────────────┐        ║
║  │  Task 4: KUBEFLOW PIPELINE (KFP) – Triggered by Airflow   │        ║
║  │  Namespace: kubeflow · EKS Cluster                         │        ║
║  │                                                            │        ║
║  │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐ │        ║
║  │  │ Component: │  │ Component:   │  │ Component:         │ │        ║
║  │  │ Train      │→ │ Evaluate     │→ │ Register Model     │ │        ║
║  │  │            │  │              │  │                    │ │        ║
║  │  │ GPU Pod    │  │ CPU Pod      │  │ CPU Pod            │ │        ║
║  │  │ p3.2xlarge │  │              │  │                    │ │        ║
║  │  │            │  │ vs Champion  │  │ MLflow Registry    │ │        ║
║  │  │ MLflow log │  │ NDCG, MRR   │  │ stage="Staging"    │ │        ║
║  │  │ params +   │  │ Evidently   │  │ tag: git_sha,      │ │        ║
║  │  │ metrics +  │  │ drift check  │  │ dataset_ver,       │ │        ║
║  │  │ artifacts  │  │ PSI, KS test │  │ training_date      │ │        ║
║  │  └────────────┘  └──────────────┘  └────────────────────┘ │        ║
║  │                       │                      │             │        ║
║  │                  Gate: PASS?              If PASS:          │        ║
║  │                  ┌────┴────┐             SNS → Slack       │        ║
║  │                 FAIL     PASS                              │        ║
║  │                  │         │                               │        ║
║  │             SNS Alert    Continue                          │        ║
║  │             Abort DAG   to Task 5                          │        ║
║  └────────────────────────────────────────────────────────────┘        ║
║                                               │                        ║
║                                               ▼                        ║
║  ┌────────────────────────────────────────────────────────────┐        ║
║  │  Task 5: BATCH INFERENCE (Spark on EKS)                    │        ║
║  │  ├── Lee modelo: MLflow stage="Production"                 │        ║
║  │  ├── Lee features: Feast offline + online                  │        ║
║  │  ├── Genera predicciones:                                  │        ║
║  │  │   ├── Recomendador: TOP-20 por usuario → Redis + S3     │        ║
║  │  │   └── Buscador: embeddings todos productos → OpenSearch │        ║
║  │  ├── Escribe: ElastiCache Redis (TTL) + Redshift (audit)   │        ║
║  │  └── Log: duration, coverage, error count → Prometheus     │        ║
║  └────────────────────────────────────────────────────────────┘        ║
║                                                                          ║
║  ALMACENAMIENTO DE ARTIFACTS                                             ║
║  ──────────────────────────────                                          ║
║  ┌────────────────────────────────────────────────────────────┐        ║
║  │  S3 Buckets                   │  MLflow Tracking Server    │        ║
║  │  ├── dsrpmart-data/raw/       │  ├── Experiments           │        ║
║  │  ├── dsrpmart-data/validated/ │  ├── Runs (params, metrics)│        ║
║  │  ├── dsrpmart-data/processed/ │  ├── Artifacts (S3 backed) │        ║
║  │  ├── dsrpmart-features/       │  └── Model Registry         │        ║
║  │  ├── dsrpmart-models/         │      ├── Staging            │        ║
║  │  └── dsrpmart-mlflow-artifacts│      ├── Production         │        ║
║  └────────────────────────────────┴───────────────────────────┘        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 10.b Arquitectura de la Solución Completa (AWS)

```
╔══════════════════════════════════════════════════════════════════════════╗
║               ARQUITECTURA AWS CLOUD-NATIVE – DSRPMart                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║                    ┌──────────────────────────────────┐                  ║
║                    │  DSRPMart Mobile App / Web App    │                  ║
║                    │  (React Native / React + Next.js) │                  ║
║                    └──────────────┬───────────────────┘                  ║
║                                   │                                      ║
║                                   ▼                                      ║
║                    ┌──────────────────────────────────┐                  ║
║                    │ Amazon API Gateway + AWS WAF       │                  ║
║                    │ (Auth: Cognito, Rate Limit, HTTPS) │                  ║
║                    └──────────┬────────┬──────────────┘                  ║
║                               │        │                                 ║
║              ┌────────────────┘        └──────────────────┐             ║
║              ▼                                            ▼             ║
║  ┌───────────────────────────┐          ┌───────────────────────────┐   ║
║  │  Recommendations Service  │          │  Search Service           │   ║
║  │  (FastAPI · EKS Pod)      │          │  (FastAPI · EKS Pod)      │   ║
║  │  ├── GET /recs/:user_id   │          │  ├── GET /search?q=...    │   ║
║  │  └── Redis lookup         │          │  ├── Query preprocess     │   ║
║  │      key: user:{id}:recs  │          │  ├── BM25+KNN retrieval   │   ║
║  └───────────┬───────────────┘          │  ├── LightGBM ranking     │   ║
║              │                          │  └── Post-processing      │   ║
║              │                          └──────────┬────────────────┘   ║
║              │                                     │                    ║
║              ▼                                     ▼                    ║
║  ┌────────────────────────────────────────────────────────────────────┐ ║
║  │                    Amazon ElastiCache (Redis 7.x)                   │ ║
║  │  ├── Cluster mode: 3 shards, 2 replicas each                       │ ║
║  │  ├── Recs:   user:{id}:recs → JSON [product_ids] TTL=6h           │ ║
║  │  ├── Search: query:{hash} → JSON [results]        TTL=30min       │ ║
║  │  └── Feast:  feast:online:{entity} → feature_vector               │ ║
║  └────────────────────────────────────────────────────────────────────┘ ║
║       ▲ (writes)                         ▲ (KNN queries)               ║
║       │                                  │                              ║
║  ┌────┴───────────────────┐     ┌────────┴──────────────────────────┐  ║
║  │ BATCH INFERENCE LAYER  │     │ Amazon OpenSearch Service          │  ║
║  │ Spark on EKS           │     │ ├── Index: products-v{N}           │  ║
║  │ (Spark Operator)       │     │ │   ├── BM25 text fields           │  ║
║  │ ├── scheduled 4x/day   │     │ │   ├── KNN vector (384d, HNSW)    │  ║
║  │ ├── reads MLflow model │     │ │   └── metadata (price,category)  │  ║
║  │ ├── reads Feast features│     │ ├── Blue-green index swap          │  ║
║  │ └── writes Redis + S3  │     │ └── 3 data nodes (r6g.xlarge)      │  ║
║  └────────────────────────┘     └───────────────────────────────────┘  ║
║       ▲                                  ▲                              ║
║       │                                  │                              ║
║  ═════╪══════════════════════════════════╪══════════════════════════   ║
║       │         ML TRAINING PLATFORM     │                              ║
║  ═════╪══════════════════════════════════╪══════════════════════════   ║
║       │                                  │                              ║
║  ┌────┴──────────────────────────────────┴───────────────────────────┐ ║
║  │                    Amazon EKS Cluster (v1.29)                      │ ║
║  │                    Managed Node Groups + Karpenter                  │ ║
║  │                                                                    │ ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ ║
║  │  │ Namespace:   │  │ Namespace:   │  │ Namespace:               │ │ ║
║  │  │  kubeflow    │  │  airflow     │  │  mlflow                  │ │ ║
║  │  │              │  │              │  │                          │ │ ║
║  │  │ KFP Pipelines│  │ MWAA Agent   │  │ Tracking Server          │ │ ║
║  │  │ Katib (HPO)  │  │ DAG sync     │  │ Model Registry           │ │ ║
║  │  │ Notebooks    │  │ from S3      │  │ Backend: RDS PostgreSQL  │ │ ║
║  │  └──────────────┘  └──────────────┘  │ Artifacts: S3             │ │ ║
║  │                                      └──────────────────────────┘ │ ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ ║
║  │  │ Namespace:   │  │ Namespace:   │  │ Namespace:               │ │ ║
║  │  │  feast       │  │  spark       │  │  monitoring              │ │ ║
║  │  │              │  │              │  │                          │ │ ║
║  │  │ Feast Server │  │ Spark Oper.  │  │ Prometheus               │ │ ║
║  │  │ Online: Redis│  │ Driver/Exec  │  │ Grafana                  │ │ ║
║  │  │ Offline: S3  │  │ Spot pools   │  │ Alertmanager             │ │ ║
║  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ ║
║  │                                                                    │ ║
║  │  ┌──────────────┐  ┌──────────────────────────────────┐           │ ║
║  │  │ Namespace:   │  │ GitOps Controller:               │           │ ║
║  │  │  serving     │  │  ArgoCD                          │           │ ║
║  │  │              │  │  ├── Syncs: dsrpmart-infra repo  │           │ ║
║  │  │ Rec Service  │  │  ├── Auto-heal on drift          │           │ ║
║  │  │ Search Svc   │  │  └── Rollback on failure         │           │ ║
║  │  └──────────────┘  └──────────────────────────────────┘           │ ║
║  └────────────────────────────────────────────────────────────────────┘ ║
║       ▲                                                                ║
║       │                                                                ║
║  ═════╪════════════════════════════════════════════════════════════   ║
║       │                   DATA LAYER                                   ║
║  ═════╪════════════════════════════════════════════════════════════   ║
║       │                                                                ║
║  ┌────┴──────────────────────────────────────────────────────────────┐ ║
║  │                                                                    │ ║
║  │  ┌──────────────────┐   ┌──────────────┐  ┌────────────────────┐  │ ║
║  │  │ Amazon S3         │   │ Amazon       │  │ Amazon Kinesis     │  │ ║
║  │  │ ├── raw/          │   │ Redshift     │  │ Data Streams       │  │ ║
║  │  │ ├── validated/    │   │ Serverless   │  │ ├── user_events    │  │ ║
║  │  │ ├── processed/    │   │              │  │ ├── search_logs    │  │ ║
║  │  │ ├── features/     │   │ DWH + Anal.  │  │ └── Firehose → S3  │  │ ║
║  │  │ └── models/       │   │ A/B analysis │  │                    │  │ ║
║  │  └──────────────────┘   └──────────────┘  └────────────────────┘  │ ║
║  │                                                                    │ ║
║  │  ┌──────────────────┐   ┌──────────────┐  ┌────────────────────┐  │ ║
║  │  │ Amazon RDS       │   │ AWS DMS       │  │ AWS Glue           │  │ ║
║  │  │ (PostgreSQL)     │   │ CDC → S3      │  │ Data Catalog       │  │ ║
║  │  │ Catálogo prod.   │   │ Catálogo sync │  │ Schema discovery   │  │ ║
║  │  └──────────────────┘   └──────────────┘  └────────────────────┘  │ ║
║  │                                                                    │ ║
║  └────────────────────────────────────────────────────────────────────┘ ║
║                                                                          ║
║  ═══════════════════════════════════════════════════════════════════    ║
║                          SECURITY LAYER                                  ║
║  ═══════════════════════════════════════════════════════════════════    ║
║  ├── IRSA: IAM Roles for Service Accounts (pods K8s)                    ║
║  ├── AWS Secrets Manager: DB creds, API keys (auto-rotation)            ║
║  ├── KMS: encryption at rest (S3, RDS, ElastiCache, Redshift)           ║
║  ├── VPC: Private subnets for EKS, NAT for egress                       ║
║  ├── WAF: Protection on API Gateway                                      ║
║  ├── ECR: Image scanning (Trivy) + cosign signing                        ║
║  └── CloudTrail: Audit log de todas las acciones AWS                     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 10.c CI/CD Despliegue de Modelo

```
╔══════════════════════════════════════════════════════════════════════════╗
║          CI/CD DESPLIEGUE DE MODELO – GitHub Actions + ArgoCD            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  DATA SCIENTIST / ML ENGINEER                                    │   ║
║  │  ├── Desarrolla en branch feature/model-improvement-v2           │   ║
║  │  ├── Experimenta en JupyterHub (Kubeflow Notebooks on EKS)      │   ║
║  │  ├── Registra runs en MLflow: experiment_name, params, metrics   │   ║
║  │  ├── DVC add: datasets versionados en S3                         │   ║
║  │  ├── git commit + git push → Pull Request a develop              │   ║
║  │  └── PR description: MLflow run_id, métricas, changelog          │   ║
║  └────────────────────────────────┬─────────────────────────────────┘   ║
║                                   │                                      ║
║                                   ▼                                      ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  GITHUB ACTIONS – CI PIPELINE (Triggered on PR)                  │   ║
║  │                                                                  │   ║
║  │  Job 1: code-quality                                             │   ║
║  │  ├── ruff check . (lint Python)                                  │   ║
║  │  ├── mypy (type checking)                                        │   ║
║  │  └── bandit (security scan Python)                                │   ║
║  │                                                                  │   ║
║  │  Job 2: tests                                                    │   ║
║  │  ├── pytest tests/unit/ --cov --cov-fail-under=80                │   ║
║  │  ├── pytest tests/integration/ (mock AWS services with moto)     │   ║
║  │  └── Great Expectations: data contract tests (sample data)       │   ║
║  │                                                                  │   ║
║  │  Job 3: reproducibility                                          │   ║
║  │  ├── dvc repro --dry (verify pipeline is reproducible)           │   ║
║  │  └── dvc diff (show data/model changes)                          │   ║
║  │                                                                  │   ║
║  │  STATUS: ✅ All checks pass → Ready for review                  │   ║
║  └────────────────────────────────┬─────────────────────────────────┘   ║
║                                   │                                      ║
║                         (PR approved + merged to develop)                ║
║                                   │                                      ║
║                                   ▼                                      ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  GITHUB ACTIONS – CD PIPELINE STAGING (on merge to develop)      │   ║
║  │                                                                  │   ║
║  │  Job 1: build-push                                               │   ║
║  │  ├── docker build --platform linux/amd64                          │   ║
║  │  ├── trivy image scan (vulnerabilities)                           │   ║
║  │  ├── docker push → Amazon ECR: {repo}:{git-sha}                  │   ║
║  │  └── cosign sign (supply chain attestation)                       │   ║
║  │                                                                  │   ║
║  │  Job 2: deploy-staging                                           │   ║
║  │  ├── Update helm/values-staging.yaml → image.tag: {git-sha}      │   ║
║  │  ├── Commit + push to dsrpmart-infra (staging branch)            │   ║
║  │  └── ArgoCD auto-syncs staging namespace                          │   ║
║  │                                                                  │   ║
║  │  Job 3: ml-pipeline-staging                                      │   ║
║  │  ├── Trigger KFP Pipeline on staging EKS cluster                 │   ║
║  │  ├── Wait for pipeline completion (timeout: 2h)                  │   ║
║  │  ├── Fetch metrics from MLflow API                                │   ║
║  │  ├── Compare vs. current Production Champion:                     │   ║
║  │  │   ├── table: NDCG, MRR, HitRate, Coverage                     │   ║
║  │  │   ├── Evidently drift report → S3                              │   ║
║  │  │   └── Auto-comment PR with comparison table                    │   ║
║  │  └── IF metrics PASS → MLflow stage = "Staging" ✅               │   ║
║  │       IF metrics FAIL → Block promotion, Slack alert ❌           │   ║
║  └────────────────────────────────┬─────────────────────────────────┘   ║
║                                   │                                      ║
║                  (Promote develop → main via Release PR)                 ║
║                                   │                                      ║
║                                   ▼                                      ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  GITHUB ACTIONS – CD PIPELINE PRODUCTION (on merge to main)      │   ║
║  │                                                                  │   ║
║  │  Job 1: production-deploy                                        │   ║
║  │  ├── Tag Docker image: :{git-sha} + :v{semver}                   │   ║
║  │  ├── Update helm/values-prod.yaml → image.tag: {git-sha}         │   ║
║  │  ├── Commit + push to dsrpmart-infra (prod branch)               │   ║
║  │  └── Create GitHub Release with changelog                         │   ║
║  │                                                                  │   ║
║  │  ┌────────────────────────────────────────────────────────┐      │   ║
║  │  │  ARGOCD (GitOps Controller on EKS)                      │      │   ║
║  │  │  ├── Detects change in dsrpmart-infra/prod               │      │   ║
║  │  │  ├── Reconciles EKS prod namespace:                      │      │   ║
║  │  │  │   ├── Rolling update pods (zero downtime)             │      │   ║
║  │  │  │   ├── readinessProbe + livenessProbe checks           │      │   ║
║  │  │  │   └── Progressive rollout (canary 10% → 50% → 100%)  │      │   ║
║  │  │  ├── IF health check fails → automatic rollback          │      │   ║
║  │  │  └── Slack: #ml-deployments "Deploy v{X} complete ✅"    │      │   ║
║  │  └────────────────────────────────────────────────────────┘      │   ║
║  │                                                                  │   ║
║  │  Job 2: post-deploy-validation                                   │   ║
║  │  ├── Smoke tests: call /recs and /search endpoints              │   ║
║  │  ├── MLflow: transition model Staging → Production               │   ║
║  │  ├── Enable shadow mode for Challenger (if applicable)           │   ║
║  │  └── Activate monitoring dashboards for new version              │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ROLLBACK AUTOMÁTICO                                                     ║
║  ─────────────────                                                       ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  Trigger:                                                        │   ║
║  │  ├── Prometheus alert: error_rate > 1% for 5 min                 │   ║
║  │  ├── Prometheus alert: latency p99 > SLA for 10 min              │   ║
║  │  └── Manual: MLOps engineer via ArgoCD UI                        │   ║
║  │                                                                  │   ║
║  │  Actions:                                                        │   ║
║  │  ├── ArgoCD: git revert values-prod.yaml → previous SHA         │   ║
║  │  ├── K8s: pods roll back to previous image                       │   ║
║  │  ├── MLflow: previous model version → stage="Production"        │   ║
║  │  ├── Redis: batch job rewrites predictions from previous model   │   ║
║  │  ├── PagerDuty: page on-call MLOps with runbook link            │   ║
║  │  └── Post-incident: blameless post-mortem within 48h            │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 11. Monitoreo, Data Drift y Observabilidad

#### Diagrama Mermaid – Loop de Monitoreo y Reentrenamiento Automático

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '13px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD
    PROD["<b>MODELO EN PRODUCCIÓN</b><br>MLflow stage = Production"]:::prod
    
    PROD --> PRED["<b>Batch Inference</b><br>Predicciones → Redis + S3"]:::inference
    PRED --> EVENTS["<b>Eventos Reales</b><br>Clicks · Compras · Sessions<br>Kinesis → Redshift"]:::data
    
    EVENTS --> DRIFT
    PROD --> DRIFT
    DRIFT["<b>Drift Monitor DAG</b><br>Airflow · Evidently AI<br>Ejecución diaria 04:00 UTC"]:::monitor
    
    DRIFT --> PSI
    PSI{"<b>Population Stability Index</b><br>¿Nivel de drift?"}:::gate
    
    PSI -->|"PSI < 0.10<br>Sin drift"| LOG["Sin acción<br>Log OK · Continuar"]:::ok
    PSI -->|"0.10 ≤ PSI < 0.25<br>Drift moderado"| WARN["Alerta Warning<br>Slack #ml-alerts"]:::warning
    PSI -->|"PSI ≥ 0.25<br>Drift crítico"| RETRAIN["<b>REENTRENAMIENTO</b><br>Trigger automático<br>Kubeflow Pipeline"]:::retrain
    
    RETRAIN --> EVAL
    EVAL{"¿Nuevo modelo<br>supera Champion?"}:::gate
    EVAL -->|"Sí"| SHADOW2["Shadow → A/B Test<br>→ Promoción"]:::success
    EVAL -->|"No"| KEEP["Mantener Champion<br>Investigar causa raíz"]:::neutral
    
    SHADOW2 --> PROD
    LOG --> PROD

    classDef prod fill:#15803d,stroke:#166534,color:#fff,stroke-width:3px,font-weight:bold
    classDef inference fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px
    classDef data fill:#6d28d9,stroke:#5b21b6,color:#fff,stroke-width:1px
    classDef monitor fill:#0e7490,stroke:#155e75,color:#fff,stroke-width:2px
    classDef gate fill:#fef3c7,stroke:#d97706,color:#92400e,stroke-width:3px,font-weight:bold
    classDef ok fill:#d1fae5,stroke:#34d399,color:#065f46,stroke-width:2px
    classDef warning fill:#fef08a,stroke:#eab308,color:#713f12,stroke-width:2px
    classDef retrain fill:#dc2626,stroke:#b91c1c,color:#fff,stroke-width:2px,font-weight:bold
    classDef success fill:#15803d,stroke:#166534,color:#fff,stroke-width:2px
    classDef neutral fill:#f1f5f9,stroke:#94a3b8,color:#475569,stroke-width:1px
```

### 11.1 Estrategia de Data Drift

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ESTRATEGIA DE DATA DRIFT                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DETECCIÓN (Evidently AI – Batch Reports)                            │
│  ──────────────────────────────────────                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Airflow DAG: data_drift_monitoring (daily 04:00 UTC)        │    │
│  │                                                              │    │
│  │  Task 1: Compute Feature Drift                               │    │
│  │  ├── Reference: training dataset (feature distributions)     │    │
│  │  ├── Current: last 24h production features (Feast offline)   │    │
│  │  ├── Tests por feature:                                      │    │
│  │  │   ├── PSI (Population Stability Index)                    │    │
│  │  │   │   └── < 0.10: sin drift │ 0.10-0.25: warning │ >0.25:│    │
│  │  │   │       alert + trigger retrain                         │    │
│  │  │   ├── KS Test (Kolmogorov-Smirnov)                       │    │
│  │  │   │   └── p-value < 0.01 → drift significativo           │    │
│  │  │   └── Wasserstein Distance (para features continuas)      │    │
│  │  └── Output: Evidently HTML report → S3 + MLflow run         │    │
│  │                                                              │    │
│  │  Task 2: Compute Prediction Drift                            │    │
│  │  ├── Distribución de scores de predicción T vs T-7           │    │
│  │  ├── PSI de distribución de Top-K product_ids                │    │
│  │  └── Alertas si > 20% de los items en Top-10 son nuevos     │    │
│  │                                                              │    │
│  │  Task 3: Compute Target Drift (delayed ground truth)         │    │
│  │  ├── CTR real vs CTR esperado (por cohort semanal)           │    │
│  │  ├── Conversion rate trend (rolling 7 days)                  │    │
│  │  └── Alerta si gap > 2 std vs media histórica              │    │
│  │                                                              │    │
│  │  Task 4: Decision                                            │    │
│  │  ├── IF max(PSI) > 0.25 → trigger immediate retrain DAG     │    │
│  │  ├── IF 0.10 < max(PSI) < 0.25 → Slack warning #ml-alerts   │    │
│  │  └── IF all PSI < 0.10 → log OK, continue normal schedule   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  MÉTRICAS MONITOREADAS POR MODELO                                    │
│  ─────────────────────────────────                                   │
│                                                                      │
│  Productos Recomendados:                                             │
│  ├── user_ctr_by_category (PSI)                                      │
│  ├── session_length (KS test)                                        │
│  ├── avg_price_viewed (Wasserstein)                                   │
│  ├── category_distribution (Chi-squared)                             │
│  └── prediction_score_distribution (PSI)                             │
│                                                                      │
│  Motor de Búsqueda:                                                  │
│  ├── query_length_distribution (KS test)                             │
│  ├── query_category_distribution (Chi-squared)                       │
│  ├── embedding_centroid_shift (cosine distance)                      │
│  ├── retrieval_score_distribution (PSI)                               │
│  └── zero_result_rate_trend (statistical process control)            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 Dashboards de Grafana

| Dashboard | Paneles Clave | Audiencia |
|---|---|---|
| **ML Model Health** | NDCG@10 por versión, PSI score por feature, modelo en producción vs staging, última fecha de retrain | Data Science + MLOps |
| **Recommendation KPIs** | CTR diario (7d rolling), Add-to-Cart rate, Revenue per Session uplift, Coverage % | Product + ML Lead |
| **Search Performance** | Latencia p50/p95/p99, Zero-result rate, Search CTR, Queries per second | Backend + MLOps |
| **Infrastructure** | CPU/Memory pods, node autoscaling events, Redis hit rate/memory, OpenSearch cluster health | ML Platform + DevOps |
| **Pipeline Health** | Airflow DAG success rate, KFP pipeline durations, failed tasks trend | MLOps |
| **Data Drift Monitor** | PSI heatmap por feature × día, trend de drift score 30 días, alerts timeline | Data Science |
| **A/B Test Results** | CTR Champion vs Challenger, Conversion lift %, p-value, sample size progress | Product + ML Lead |
| **Cost Optimization** | Spend por servicio AWS, cost per prediction, Spot vs On-Demand usage | ML Lead + Management |

### 11.3 Alerting Rules

```yaml
# Prometheus AlertManager rules (EKS)
groups:
  - name: ml-model-alerts
    rules:
      - alert: ModelLatencyHigh
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{service="recs-service"}[5m])) > 0.01
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "Recommendations API p99 latency > 10ms"
          runbook: "https://wiki.dsrpmart.internal/runbooks/recs-latency"

      - alert: SearchLatencyCritical
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{service="search-service"}[5m])) > 0.1
        for: 5m
        labels: { severity: critical }
        annotations:
          summary: "Search API p95 latency > 100ms SLA breach"

      - alert: DataDriftDetected
        expr: data_drift_psi_max > 0.25
        for: 1m
        labels: { severity: warning }
        annotations:
          summary: "Data drift PSI > 0.25 detected, retraining triggered"

      - alert: BatchInferenceFailed
        expr: airflow_dag_run_state{dag_id=~".*recommender.*", state="failed"} > 0
        labels: { severity: critical }
        annotations:
          summary: "Batch inference DAG failed"

      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.85
        for: 10m
        labels: { severity: warning }
        annotations:
          summary: "ElastiCache Redis memory usage > 85%"
```

---

## 12. Fuentes y Bibliografía

### 12.1 Papers Académicos y Técnicos

| # | Referencia | Descripción | URL |
|---|---|---|---|
| 1 | Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). *Machine Learning Operations (MLOps): Overview, Definition, and Architecture.* IEEE Access. | Paper fundacional que define los niveles de madurez MLOps 0-2 utilizados en este documento | https://arxiv.org/abs/2205.02302 |
| 2 | Covington, P., Adams, J., & Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations.* RecSys. | Arquitectura Two-Tower original de Google que inspira nuestro Stage A de retrieval | https://research.google/pubs/pub45530/ |
| 3 | Yi, X. et al. (2019). *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations.* RecSys. | In-batch sampled softmax y corrección de bias que usamos en el Two-Tower | https://research.google/pubs/pub48840/ |
| 4 | Burges, C.J.C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research. | Fundamento teórico de LambdaRank implementado vía LightGBM en ambos modelos | https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/ |
| 5 | Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. | Base del modelo Sentence-BERT utilizado para embeddings de búsqueda | https://arxiv.org/abs/1908.10084 |
| 6 | Barkan, O. & Koenigstein, N. (2016). *Item2Vec: Neural Item Embedding for Collaborative Filtering.* IEEE MLSP. | Método de embeddings Item2Vec aplicado a secuencias de sesión | https://arxiv.org/abs/1603.04259 |
| 7 | Carbonell, J. & Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR. | Maximal Marginal Relevance para diversificación de resultados en re-ranking | https://dl.acm.org/doi/10.1145/290941.291025 |
| 8 | Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. | Anti-patrones de ML en producción que esta arquitectura evita | https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html |

### 12.2 Libros de Referencia

| # | Libro | Autor(es) | Relevancia |
|---|---|---|---|
| 1 | *Designing Machine Learning Systems* | Chip Huyen (O'Reilly, 2022) | Arquitectura ML end-to-end, feature stores, data distribution shifts, testing |
| 2 | *Machine Learning Engineering* | Andriy Burkov (2020) | Best practices de ML en producción, model management, deployment patterns |
| 3 | *Reliable Machine Learning* | Cathy Chen et al. (O'Reilly, 2022) | MLOps, monitoring, CI/CD para ML, incident management |
| 4 | *Building Machine Learning Pipelines* | Hannes Hapke & Catherine Nelson (O'Reilly, 2020) | TFX, Kubeflow Pipelines, Airflow para ML |
| 5 | *Practical Recommender Systems* | Kim Falk (Manning, 2019) | Sistemas de recomendación: collaborative filtering, content-based, hybrid |
| 6 | *Kubernetes in Action* | Marko Lukša (Manning, 2nd Ed 2024) | Fundamentos de K8s, deployments, pods, servicios, usado para EKS |
| 7 | *Terraform: Up & Running* | Yevgeniy Brikman (O'Reilly, 3rd Ed 2022) | IaC con Terraform, módulos, state management, CI/CD para infra |

### 12.3 Documentación Oficial de Herramientas

| # | Herramienta | URL |
|---|---|---|
| 1 | MLflow Documentation | https://mlflow.org/docs/latest/index.html |
| 2 | Kubeflow Pipelines on AWS EKS | https://www.kubeflow.org/docs/distributions/aws/ |
| 3 | Apache Airflow – Amazon MWAA | https://docs.aws.amazon.com/mwaa/ |
| 4 | Feast Feature Store | https://docs.feast.dev/ |
| 5 | Evidently AI (Data Drift) | https://docs.evidentlyai.com/ |
| 6 | Amazon EKS Best Practices | https://aws.github.io/aws-eks-best-practices/ |
| 7 | ArgoCD GitOps for K8s | https://argo-cd.readthedocs.io/ |
| 8 | Amazon OpenSearch KNN Plugin | https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html |
| 9 | Sentence-Transformers Library | https://www.sbert.net/ |
| 10 | LightGBM (Learning-to-Rank) | https://lightgbm.readthedocs.io/en/stable/ |
| 11 | Spark on Kubernetes (Spark Operator) | https://github.com/kubeflow/spark-operator |
| 12 | Terraform AWS Provider | https://registry.terraform.io/providers/hashicorp/aws/latest/docs |
| 13 | Karpenter (K8s Autoscaler) | https://karpenter.sh/docs/ |
| 14 | Great Expectations (Data Quality) | https://docs.greatexpectations.io/ |
| 15 | FAISS (Facebook AI Similarity Search) | https://github.com/facebookresearch/faiss |
| 16 | DVC – Data Version Control | https://dvc.org/doc |
| 17 | Amazon Kinesis Data Streams | https://docs.aws.amazon.com/streams/latest/dev/ |
| 18 | Amazon Redshift Serverless | https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-whatis.html |
| 19 | Prometheus + Grafana Stack | https://prometheus.io/docs/ / https://grafana.com/docs/ |
| 20 | Trivy (Container Security) | https://trivy.dev/latest/docs/ |

### 12.4 Blogs y Recursos Técnicos de Industria

| # | Referencia | URL |
|---|---|---|
| 1 | Model Cards for Model Reporting (Google) | https://modelcards.withgoogle.com/ |
| 2 | Kaggle – Model Cards Template | https://www.kaggle.com/code/var0101/model-cards |
| 3 | ML Design Docs – Eugene Yan | https://eugeneyan.com/writing/ml-design-docs/ |
| 4 | Software Engineering RFC and Design – Pragmatic Engineer | https://newsletter.pragmaticengineer.com/p/software-engineering-rfc-and-design |
| 5 | Champion/Challenger Pattern for ML Deployment | https://christophergs.com/machine%20learning/2019/03/30/deploying-and-versioning-data-science-models-at-scale/ |
| 6 | Spotify – ML Infrastructure Case Study | https://engineering.atspotify.com/2019/12/13/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow/ |
| 7 | Netflix – System Architectures for Personalization and Recommendation | https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8 |
| 8 | Uber – Michelangelo ML Platform | https://www.uber.com/blog/michelangelo-machine-learning-platform/ |
| 9 | Airbnb – Machine Learning Infrastructure | https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d |
| 10 | AWS Well-Architected ML Lens | https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html |

---

*Documento elaborado como Proyecto Final del Curso IV – Especialización Machine Learning Engineering · Febrero 2026*  
*Herramientas recomendadas para pasar los diagramas de texto a formato visual: [Excalidraw](https://excalidraw.com/) · [draw.io](https://draw.io/) · [tldraw](https://www.tldraw.com/) · [Miro](https://miro.com/) · [Lucidchart](https://lucidchart.com/)*
