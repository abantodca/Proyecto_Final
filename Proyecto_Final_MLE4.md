# Proyecto Final – Especialización Machine Learning Engineering · Curso IV

**Autor:** Carlos Abanto  
**Fecha:** Febrero 2026  
**Cliente:** DSRPMart – Marketplace digital en Latinoamérica  
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
   - 1.1 [Contexto de Negocio — DSRPMart](#11-contexto-de-negocio--dsrpmart)
   - 1.2 [¿Por Qué Machine Learning para una Aplicación de Ventas?](#12-por-qué-machine-learning-para-una-aplicación-de-ventas)
   - 1.3 [Propuesta de Solución](#13-propuesta-de-solución)
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

### 1.1 Contexto de Negocio — DSRPMart

**DSRPMart** es una startup de marketplace digital en Latinoamérica que conecta vendedores (sellers) con compradores a través de una aplicación móvil y web. El modelo de negocio se basa en comisiones por transacción, publicidad interna (productos sponsoreados) y suscripciones premium para sellers.

| Indicador de Negocio | Situación Actual | Desafío |
|---|---|---|
| **Catálogo** | ~500K productos activos de miles de sellers | Los usuarios no encuentran lo que necesitan entre la oferta masiva |
| **Usuarios** | ~10M usuarios registrados, ~2M activos mensuales | Baja conversión: los usuarios ven muchos productos pero compran pocos |
| **Tasa de conversión** | ~1.8% (benchmark e-commerce Latam: 2.5-3.5%) | Brecha significativa frente a competidores con ML maduros |
| **Búsquedas sin resultado** | ~8% de las consultas | Pérdida directa de ventas y frustración del usuario |
| **Ticket promedio** | $35 USD | Oportunidad de aumentar con cross-sell y recomendaciones relevantes |
| **Tiempo en app** | 4.2 min por sesión | Inferior al benchmark de apps con personalización ML (6-8 min) |

**El problema fundamental:** DSRPMart tiene un catálogo amplio y una base de usuarios en crecimiento, pero **carece de inteligencia artificial para conectar al usuario correcto con el producto correcto en el momento correcto**. Esto se traduce en baja conversión, baja retención y pérdida de revenue frente a competidores que ya incorporan ML (Mercado Libre, Amazon, Falabella).

### 1.2 ¿Por Qué Machine Learning para una Aplicación de Ventas?

La propuesta de incorporar ML/IA no es un ejercicio tecnológico sino una **necesidad competitiva con impacto directo en los ingresos**:

| Problema de Negocio | Solución con ML/IA | Impacto Esperado en Ventas |
|---|---|---|
| Usuario no encuentra productos relevantes entre 500K SKUs | **Recomendaciones personalizadas** (Two-Tower NN + LambdaRank) que aprenden del comportamiento de cada usuario | +15-25% CTR en sección "Para Ti" → +8% Revenue per Session |
| Búsqueda devuelve resultados irrelevantes o vacíos | **Motor de Búsqueda inteligente** (Sentence-BERT + LightGBM Ranking) con comprensión semántica | -75% zero-result rate → +10% Search Conversion |
| Recomendaciones estáticas ("más vendidos") para todos | Rankings personalizados actualizados 4x/día por usuario | +12% Tiempo en App → mayor engagement y retención |
| Sin capacidad de medir impacto de cambios | **A/B Testing automatizado** con métricas de negocio (CTR, conversion, revenue) | Decisiones data-driven, no por intuición |
| Modelos manuales que se degradan con el tiempo | **MLOps automatizado** con detección de drift y reentrenamiento continuo | Modelos siempre actualizados → revenue sostenido |

> **Retorno de inversión estimado:** Según estudios de McKinsey (2023), la personalización aumenta los ingresos entre 10-15%. Proyectamos que esta inversión en ML generará un incremento de **10-15% en ventas totales (GMV)** en los primeros 12 meses, con recuperación de la inversión en 6-8 meses considerando los costos de infraestructura AWS y el equipo de 11 personas.

### 1.3 Propuesta de Solución

Proponemos implementar una plataforma de **Machine Learning en producción** sobre **Amazon Web Services (AWS)**, que permita entrenar, desplegar y actualizar modelos de inteligencia artificial de forma automatizada. Esta plataforma resolverá los dos problemas con mayor impacto en ventas:

| Caso de Uso | Qué resuelve | Frecuencia | Impacto en ventas |
|---|---|---|---|
| **Productos Recomendados** | "¿Qué productos le interesan a ESTE usuario?" | Batch 4x/día + serving < 5ms | Aumento de conversión y ticket promedio |
| **Motor de Búsqueda Inteligente** | "¿Qué productos coinciden con ESTA búsqueda?" | Indexación batch + serving < 100ms | Reducción de búsquedas sin resultado → más ventas |

La arquitectura propuesta garantiza:

- **Impacto en ventas medible** mediante A/B testing riguroso con KPIs de negocio (CTR, conversión, revenue)
- **Automatización CI/CD** completa desde el commit hasta producción, con rollback automático
- **Modelos siempre actualizados** gracias a Continuous Training activado por data drift
- **Escalabilidad** horizontal para soportar el crecimiento de usuarios y catálogo
- **Costos optimizados** con Spot Instances (~70% ahorro) y arquitectura serverless donde corresponda

Ambos casos comparten infraestructura, pipelines CI/CD y monitoring, maximizando la reutilización y reduciendo costos operativos.

---

## 2. Matriz de Cumplimiento de Requerimientos

La siguiente tabla mapea cada requerimiento del proyecto con la sección del documento donde se desarrolla, asegurando **completitud y trazabilidad** para la evaluación.

| # | Requerimiento del Proyecto | Dónde se resuelve | Qué se entrega |
|:---:|---|---|---|
| **1** | **Flujo E2E por caso de uso:** algoritmos, fuentes de datos, optimizaciones, Model Card, Diccionario de Datos, Métricas de Negocio | **Sección 4** — Productos Recomendados *(subsecciones 4.2 a 4.6)* y **Sección 5** — Motor de Búsqueda *(subsecciones 5.2 a 5.6)* | Diagrama E2E, tabla de algoritmos con justificación, Model Card completa, catálogo de fuentes, KPIs de negocio |
| **2** | **Tipo de Solución:** batch, real-time o streaming con argumentación | **Sección 6** — Tipo de Solución | Tabla comparativa por criterio, argumentación de descarte de alternativas |
| **3** | **Stack Tecnológico:** version control, cloud, IaC, model management, orquestación, librerías, CI/CD, métricas/monitoring, adicionales | **Sección 7** — Stack Tecnológico AWS *(subsecciones 7.a a 7.h)* + **Sección 7.i** — Análisis Comparativo | Tabla por categoría, config YAML, código Python, **matriz de decisión con alternativas evaluadas** |
| **4** | **Estrategia de despliegue:** shadow, backtest, champion-challenger, A/B test con diagrama de proceso | **Sección 8** — Estrategia de Despliegue | Diagrama de 5 fases, flujo Champion/Challenger → Shadow → A/B, criterios estadísticos |
| **5** | **Pasos de construcción:** desarrollos, actores/equipos, colaboración | **Sección 9** — Pasos de Construcción | Roadmap por Sprint (16 sem/caso), organigrama, modelo de colaboración Scrum adaptado |
| **6** | **Diagramas de arquitectura:** E2E training, arquitectura de solución, CI/CD de despliegue de modelo | **Sección 10** — Diagramas de Arquitectura *(subsecciones 10.a a 10.c)* | 3 diagramas detallados: Pipeline E2E, Arquitectura AWS completa, CI/CD GitOps |

> **Nota:** La estructura del documento sigue un orden lógico: primero el **POR QUÉ** (Sección 1 — contexto de negocio y justificación de ML), luego el **QUÉ** (Secciones 4 y 5 — casos de uso) y finalmente el **CÓMO** (Secciones 6–10 — stack, despliegue, arquitectura). Cada decisión técnica está respaldada por su impacto en las métricas de ventas de DSRPMart.

---

## 3. Nivel de Madurez MLOps

MLOps (Machine Learning Operations) define qué tan automatizado está el proceso de entrenar, desplegar y mantener modelos de ML en producción. Existen 3 niveles: en el **Nivel 0** todo es manual; en el **Nivel 1** el entrenamiento está automatizado pero el despliegue es manual; en el **Nivel 2** todo el ciclo está automatizado, incluyendo despliegue y reentrenamiento ante cambios en los datos.

Esta propuesta implementa **MLOps Nivel 2 (Automatización completa + CI/CD)** según la clasificación académica de [Kreuzberger et al., 2023](https://arxiv.org/abs/2205.02302). A continuación se muestra la progresión entre niveles y por qué el Nivel 2 es el adecuado para DSRPMart.

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

**¿Por qué NO Nivel 3 (Automatización Total)?** No es necesario aún porque DSRPMart opera solo 2 modelos con un equipo de 11 personas. El Nivel 3 agrega complejidad (pipelines que se auto-reparan, selección automática de modelos) sin beneficio proporcional a esta escala. Se puede evolucionar a Nivel 3 en el futuro cuando haya más modelos en producción.

---

## 4. Caso 1 – Productos Recomendados

### 4.1 Descripción del Problema

Cuando un usuario abre la app de DSRPMart, actualmente ve una lista genérica de "productos populares" que es idéntica para todos. Esto ignora que cada persona tiene gustos e intereses distintos según su historial de navegación, compras y búsquedas. El resultado: los usuarios hacen pocos clicks (**menos del 5%**) en la sección de inicio y se pierden muchas oportunidades de venta.

**Solución propuesta:** Generar un **ranking personalizado de 20 productos** para cada usuario, actualizado 4 veces al día. El sistema aprende del comportamiento de cada persona (qué mira, qué compra, qué busca) para mostrarle lo que realmente le interesa. En lugar de "una tienda igual para todos", cada usuario verá "su tienda personalizada".

### 4.2 Flujo End-to-End

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD

    %% ── Class definitions ──
    classDef aws fill:#FF9900,stroke:#232F3E,color:#232F3E,rx:6
    classDef data fill:#3B82F6,stroke:#1E40AF,color:#fff,rx:6
    classDef model fill:#8B5CF6,stroke:#5B21B6,color:#fff,rx:6
    classDef tool fill:#10B981,stroke:#047857,color:#fff,rx:6
    classDef gate fill:#EF4444,stroke:#991B1B,color:#fff,rx:6
    classDef serve fill:#F59E0B,stroke:#92400E,color:#232F3E,rx:6
    classDef store fill:#06B6D4,stroke:#0E7490,color:#fff,rx:6

    %% ═══════════════════════════════════════════════════════════
    %% FUENTES DE DATOS
    %% ═══════════════════════════════════════════════════════════
    subgraph FUENTES["<b>FUENTES DE DATOS</b>"]
        direction LR
        A["<b>Kinesis Data Streams</b><br/>App events — clicks, views, cart"]:::aws
        A1["<b>Kinesis Firehose</b><br/>→ S3 raw/events/ cada 5 min"]:::aws
        B["<b>RDS PostgreSQL</b><br/>Catálogo productos · CDC"]:::aws
        B1["<b>AWS DMS</b><br/>→ S3 raw/catalog/ incremental"]:::aws
        C["<b>Redshift DWH</b><br/>Historial compras / transacciones"]:::aws
        C1["<b>UNLOAD</b><br/>→ S3 raw/transactions/ diario"]:::aws
        D["<b>ElastiCache Redis</b><br/>Sesión activa · features real-time"]:::aws
    end

    A ==> A1
    B ==> B1
    C ==> C1

    %% ═══════════════════════════════════════════════════════════
    %% 1 · DATA INGESTION & VALIDATION
    %% ═══════════════════════════════════════════════════════════
    subgraph VALIDATION["<b>1 · DATA INGESTION &amp; VALIDATION</b>"]
        V1["<b>Great Expectations</b><br/>Schema · nulls · ranges · freshness"]:::tool
        V2["<b>S3 validated/</b><br/>Parquet particionado por fecha"]:::data
        V3["<b>SNS → Slack</b><br/>Alerta si validación falla"]:::serve
    end

    A1 & B1 & C1 ==> V1
    V1 ==> V2
    V1 -.-> V3

    %% ═══════════════════════════════════════════════════════════
    %% 2 · FEATURE ENGINEERING
    %% ═══════════════════════════════════════════════════════════
    subgraph FEATURES["<b>2 · FEATURE ENGINEERING</b><br/>Spark on EKS"]
        F1["<b>User Features</b><br/>CTR por categoría 7/14/30 d<br/>Frecuencia compra · RFM score<br/>Session embedding<br/>Hora / día cíclicas sin-cos"]:::data
        F2["<b>Item Features</b><br/>Item2Vec embedding<br/>Popularidad 24 h<br/>Precio · categoría L1-L3<br/>Stock · margen · freshness"]:::data
        F3["<b>Cross Features</b><br/>User × Category affinity<br/>Cosine sim user_emb · item_emb"]:::data
        F4["<b>Feast Feature Store</b><br/>Offline → S3 · Online → Redis"]:::store
        F5["<b>DVC</b><br/>Versiona dataset en S3"]:::tool
    end

    V2 ==> F1 & F2 & F3
    D -.-> F1
    F1 & F2 & F3 ==> F4
    F1 & F2 & F3 -.-> F5

    %% ═══════════════════════════════════════════════════════════
    %% 3 · MODEL TRAINING
    %% ═══════════════════════════════════════════════════════════
    subgraph TRAINING["<b>3 · MODEL TRAINING</b><br/>KFP · EKS GPU p3.2xlarge"]
        T1["<b>Stage A — Retrieval</b><br/>Two-Tower NN · TensorFlow<br/>User Tower 256→128→64<br/>Item Tower 256→128→64<br/>In-batch sampled softmax<br/>→ embeddings dim 64"]:::model
        T2["<b>ANN Index</b><br/>FAISS / OpenSearch KNN"]:::store
        T3["<b>Stage B — Ranking</b><br/>LightGBM LambdaRank<br/>Features: user + item + cross + retrieval<br/>Optimiza NDCG@10<br/>Optuna 50 trials"]:::model
        T4["<b>Stage C — Re-Ranking</b><br/>MMR diversidad<br/>λ = 0.7 relevance vs diversity"]:::model
        T5["<b>MLflow Tracking</b><br/>log_params · log_metrics<br/>log_artifact · SHA tag"]:::tool
    end

    F4 ==> T1
    T1 ==> T2
    T1 ==> T3
    T3 ==> T4
    T1 & T3 & T4 -.-> T5

    %% ═══════════════════════════════════════════════════════════
    %% 4 · MODEL EVALUATION
    %% ═══════════════════════════════════════════════════════════
    subgraph EVALUATION["<b>4 · MODEL EVALUATION</b>"]
        E1["<b>Métricas Offline vs Champion</b><br/>NDCG@10 · NDCG@20<br/>Hit Rate@10 · MAP@10<br/>MRR · Catalog Coverage"]:::data
        E2["<b>Evidently AI — Drift</b><br/>PSI por feature<br/>KS Test scores<br/>Target drift clicks / compras"]:::tool
        E3{"<b>Decision Gate</b><br/>NDCG@10 &gt; champ − 0.02<br/>AND PSI &lt; 0.25"}:::gate
        E4["<b>Evidently Report</b><br/>HTML → S3 + MLflow"]:::tool
    end

    T4 ==> E1
    T4 ==> E2
    E1 & E2 ==> E3
    E2 -.-> E4

    %% ═══════════════════════════════════════════════════════════
    %% 5 · MODEL REGISTRATION
    %% ═══════════════════════════════════════════════════════════
    subgraph REGISTRATION["<b>5 · MODEL REGISTRATION</b>"]
        R1["<b>MLflow Model Registry</b><br/>create_model_version<br/>Stage: Staging"]:::tool
        R2["<b>Tags</b><br/>git_sha · dataset_version DVC<br/>training_date"]:::data
        R3["<b>Artifacts</b><br/>TF SavedModel + LightGBM .pkl<br/>→ S3 artifact store"]:::data
        R4["<b>SNS → Slack</b><br/>#ml-models"]:::serve
    end

    E3 -- PASS --> R1
    R1 -.-> R2
    R1 -.-> R3
    R1 ==> R4

    %% ═══════════════════════════════════════════════════════════
    %% 6 · BATCH INFERENCE
    %% ═══════════════════════════════════════════════════════════
    subgraph BATCH["<b>6 · BATCH INFERENCE</b><br/>Spark on EKS · 4×/día"]
        BI1["<b>Load modelo</b><br/>MLflow stage = Production"]:::model
        BI2["<b>Load features</b><br/>Feast offline + online"]:::store
        BI3["<b>Genera TOP-20</b><br/>Ranking por usuario activo"]:::model
        BI4["<b>Redis ElastiCache</b><br/>user:id:recs TTL = 6 h"]:::serve
        BI5["<b>S3 + Redshift</b><br/>predictions/ auditoría"]:::data
        BI6["<b>Métricas batch</b><br/>Cobertura · p50/p99 latencia"]:::tool
    end

    R1 ==> BI1
    BI1 ==> BI2
    BI2 ==> BI3
    BI3 ==> BI4
    BI3 ==> BI5
    BI3 -.-> BI6

    %% ── Subgraph styling ──
    style FUENTES fill:#fef3c7,stroke:#f59e0b,rx:10,color:#78350f
    style VALIDATION fill:#dbeafe,stroke:#3b82f6,rx:10,color:#1e3a5f
    style FEATURES fill:#ede9fe,stroke:#8b5cf6,rx:10,color:#3b0764
    style TRAINING fill:#fce7f3,stroke:#ec4899,rx:10,color:#831843
    style EVALUATION fill:#fee2e2,stroke:#ef4444,rx:10,color:#7f1d1d
    style REGISTRATION fill:#d1fae5,stroke:#10b981,rx:10,color:#064e3b
    style BATCH fill:#fff7ed,stroke:#f97316,rx:10,color:#7c2d12
```

### 4.3 Algoritmos y Justificación

El sistema de recomendación funciona en 3 etapas, como un embudo: primero selecciona candidatos del catálogo completo, luego los ordena por relevancia, y finalmente diversifica para no mostrar solo productos del mismo tipo.

| Etapa | Algoritmo | Qué hace (en palabras simples) | Por qué este algoritmo |
|---|---|---|---|
| **1. Selección de candidatos** | **Two-Tower Neural Network** (TensorFlow) | Dos redes neuronales: una "entiende" al usuario y otra "entiende" los productos. Encuentra los 100 productos más afines a cada usuario | Permite buscar entre millones de productos en milisegundos usando vectores matemáticos pre-calculados |
| **2. Ranking** | **LightGBM LambdaRank** | Toma los 100 candidatos y los ordena de mejor a peor según múltiples señales (historial, precio, popularidad, categoría) | Optimiza directamente la calidad del orden (NDCG), es rápido y permite interpretar qué factores influyeron |
| **3. Diversificación** | **MMR (Maximal Marginal Relevance)** | Evita que el top-20 sea todo del mismo tipo (ej: solo zapatos). Mezcla relevancia con variedad | No requiere reentrenamiento, solo un parámetro ajustable |
| *Auxiliar* | **Item2Vec** | Aprende qué productos se ven juntos en una sesión (similar a cómo se aprenden relaciones entre palabras) | Genera representaciones numéricas que capturan similitud real entre productos |
| *Cold Start* | **Popularidad por segmento** | Para usuarios nuevos sin historial: muestra los más populares de su región/categoría | Solución rápida sin necesidad de datos previos del usuario |

**Optimizaciones aplicadas** (reducen costos y mejoran velocidad):

- **Mixed Precision Training** — Entrena el modelo con menor precisión numérica, lo que lo hace 2 veces más rápido en GPU sin perder calidad
- **Feature hashing** — Comprime categorías de productos para manejar eficientemente +10.000 tipos
- **Negative sampling adaptativo** — El modelo aprende no solo de lo que el usuario compró, sino también de lo que NOT le interesó, mejorando la precisión
- **Incremental training** — En vez de reentrenar desde cero cada día, continua desde el modelo anterior (más rápido y económico)

### 4.4 Model Card – Productos Recomendados

> Una **Model Card** es la ficha técnica del modelo de ML — documenta qué es, cómo fue entrenado, qué tan bien funciona, y cuándo se debe re-evaluar. Es similar a la ficha técnica de un medicamento: permite auditar y tomar decisiones informadas.

> **MODEL CARD — DSRPMart Product Recommender v2.0**

**Información General**

| Atributo | Detalle |
|---|---|
| **Nombre del modelo** | `product_recommender` |
| **Versión** | 2.0.0 |
| **Tipo** | Two-Tower NN (Retrieval) + LambdaRank (Ranking) |
| **Framework** | TensorFlow 2.15 + LightGBM 4.3 |
| **Propietario** | Equipo Data Science – DSRPMart |
| **Fecha creación** | Febrero 2026 |
| **Revisado por** | ML Lead / MLOps Lead |
| **Frecuencia retrain** | Diario (incremental) + Semanal (full retrain) |

**Datos de Entrenamiento**

| Atributo | Detalle |
|---|---|
| **Período** | Últimas 12 semanas (rolling window) |
| **Volumen** | ~150M eventos de interacción / ~10M usuarios |
| **Fuente principal** | S3 `s3://dsrpmart-data/processed/events/` |
| **Split estrategia** | Temporal – Train (semanas 1-9) / Val (10-11) / Test (12). NO random split. |
| **Preprocesamiento** | Spark on EKS → Feature Store Feast |

**Métricas de Evaluación (Offline — Test Set)**

| Métrica | Valor |
|---|---|
| NDCG@10 | 0.391 |
| NDCG@20 | 0.347 |
| Hit Rate@10 | 0.624 |
| MAP@10 | 0.218 |
| MRR | 0.302 |
| Catalog Coverage | 71% (productos distintos en recs) |
| Retrieval Recall@100 | 0.87 (Two-Tower → top 100 candidates) |

**Métricas de Negocio Impactadas**

- CTR (Click-Through Rate) en sección "Para Ti"
- Add-to-Cart Rate desde recomendaciones
- Revenue per Session (uplift vs sin recomendaciones)
- Engagement: Tiempo promedio en app por sesión
- Retention: D7 retention rate de usuarios activos

**Limitaciones y Sesgos Conocidos**

- Usuarios con < 5 interacciones usan fallback de popularidad
- Posible popularity bias: mitigado con MMR (diversity λ=0.7)
- Rankings > 6h de antigüedad pueden no reflejar stock actualizado
- No captura tendencias de minutos (ej: flash sale viral) sin streaming

**Uso Previsto**

- Generación batch de TOP-20 productos personalizados, actualizados 4 veces al día (00:00, 06:00, 12:00, 18:00 UTC). Servido vía Redis ElastiCache con latencia < 5ms desde la API.

**Umbrales de Alerta (Automated Guardrails)**

| Umbral | Acción |
|---|---|
| NDCG@10 offline < 0.33 | Bloquear despliegue |
| CTR online < 0.07 | Activar análisis de causa raíz |
| PSI cualquier feature > 0.25 | Trigger reentrenamiento urgente |
| Coverage < 50% | Revisar pipeline de candidatos |
| Latencia Redis p99 > 10ms | Escalar ElastiCache |

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

En una aplicación de ventas con más de 500.000 productos, la **búsqueda es el canal principal de conversión**: cuando un usuario busca algo, tiene intención clara de comprar. El problema es que el buscador actual de DSRPMart solo encuentra coincidencias de palabras exactas, lo que genera tres problemas concretos:

- **8% de búsquedas no devuelven ningún resultado** — por ejemplo, si el usuario escribe "celular barato" pero el producto se llama "smartphone económico", el buscador no los conecta
- **Resultados poco relevantes** — se ordenan por fecha de publicación, no por lo que le interesa al usuario
- **Sin personalización** — todos los usuarios ven los mismos resultados para la misma búsqueda

**Solución propuesta:** Un buscador inteligente que **entiende el significado** de lo que el usuario escribe (no solo las palabras exactas) y ordena los resultados según la relevancia para cada persona. Así, "celular" encontrará "smartphone" y los resultados se adaptarán a las preferencias de cada usuario.

### 5.2 Flujo End-to-End

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD

    %% ── Class definitions ──
    classDef aws fill:#FF9900,stroke:#232F3E,color:#232F3E,rx:6
    classDef data fill:#3B82F6,stroke:#1E40AF,color:#fff,rx:6
    classDef model fill:#8B5CF6,stroke:#5B21B6,color:#fff,rx:6
    classDef tool fill:#10B981,stroke:#047857,color:#fff,rx:6
    classDef gate fill:#EF4444,stroke:#991B1B,color:#fff,rx:6
    classDef serve fill:#F59E0B,stroke:#92400E,color:#232F3E,rx:6
    classDef store fill:#06B6D4,stroke:#0E7490,color:#fff,rx:6
    classDef query fill:#EC4899,stroke:#9D174D,color:#fff,rx:6

    %% ═══════════════════════════════════════════════════════════
    %% FUENTES DE DATOS
    %% ═══════════════════════════════════════════════════════════
    subgraph FUENTES2["<b>FUENTES DE DATOS</b>"]
        direction LR
        SA["<b>Catálogo productos</b><br/>RDS → DMS → S3 CDC"]:::aws
        SB["<b>Historial búsquedas</b><br/>Kinesis → S3"]:::aws
        SC["<b>Ground truth</b><br/>Queries con resultados clickeados"]:::data
        SD["<b>Sinónimos / Expansiones</b><br/>Curado por equipo search · Git"]:::data
    end

    %% ═══════════════════════════════════════════════════════════
    %% 1 · DATA PREPARATION
    %% ═══════════════════════════════════════════════════════════
    subgraph DATAPREP["<b>1 · DATA PREPARATION</b><br/>Airflow MWAA"]
        DP1["<b>Build training pairs</b><br/>query · product_clicked"]:::data
        DP2["<b>Negative sampling</b><br/>Mostrados pero NO clickeados"]:::data
        DP3["<b>Great Expectations</b><br/>Schema · volumen mínimo"]:::tool
        DP4["<b>S3 processed/</b><br/>search_training/"]:::data
    end

    SA & SB & SC ==> DP1
    DP1 ==> DP2
    DP2 ==> DP3
    DP3 ==> DP4

    %% ═══════════════════════════════════════════════════════════
    %% 2 · EMBEDDINGS GENERATION
    %% ═══════════════════════════════════════════════════════════
    subgraph EMBEDDINGS["<b>2 · EMBEDDINGS GENERATION</b><br/>KFP · EKS GPU"]
        EM1["<b>Sentence-BERT fine-tune</b><br/>Base: all-MiniLM-L6-v2 384 d<br/>Fine-tune query ↔ título + desc<br/>Contrastive Loss + hard negatives"]:::model
        EM2["<b>Product Embeddings</b><br/>Generar embeddings de<br/>TODO el catálogo"]:::model
        EM3["<b>MLflow</b><br/>log model · params · eval metrics<br/>S3 embeddings/"]:::tool
    end

    DP4 ==> EM1
    EM1 ==> EM2
    EM2 ==> EM3
    EM1 -.-> EM3

    %% ═══════════════════════════════════════════════════════════
    %% 3 · INDEX CONSTRUCTION
    %% ═══════════════════════════════════════════════════════════
    subgraph INDEX["<b>3 · INDEX CONSTRUCTION</b><br/>KFP Component"]
        IX1["<b>OpenSearch KNN</b><br/>HNSW algorithm<br/>Alt: FAISS IVF-PQ en S3"]:::store
        IX2["<b>Catálogo completo</b><br/>product_id · title · category<br/>price · stock · embedding"]:::data
        IX3["<b>Index blue-green</b><br/>products-vN"]:::store
    end

    EM2 ==> IX1
    IX1 ==> IX2
    IX2 ==> IX3

    %% ═══════════════════════════════════════════════════════════
    %% 4 · RANKING MODEL
    %% ═══════════════════════════════════════════════════════════
    subgraph RANKING["<b>4 · RANKING MODEL</b><br/>KFP · EKS"]
        RK1["<b>LightGBM LambdaRank</b><br/>Learning-to-Rank"]:::model
        RK2["<b>Features</b><br/>BM25 score · Semantic sim<br/>Exact match · Popularidad CTR<br/>User-item affinity<br/>Price competitiveness · Stock"]:::data
        RK3["<b>Labels</b><br/>click = 1 · no-click = 0 · purchase = 2<br/>Optimiza NDCG@10"]:::data
        RK4["<b>MLflow Registry</b><br/>Staging"]:::tool
        RK5{"<b>Decision Gate</b>"}:::gate
    end

    DP4 ==> RK1
    RK2 -.-> RK1
    RK3 -.-> RK1
    RK1 ==> RK5
    RK5 -- PASS --> RK4

    %% ═══════════════════════════════════════════════════════════
    %% 5 · DAILY RE-INDEX
    %% ═══════════════════════════════════════════════════════════
    subgraph REINDEX["<b>5 · DAILY RE-INDEX</b><br/>Airflow 02:00 UTC"]
        RI1["<b>Recompute embeddings</b><br/>Productos nuevos / modificados CDC"]:::model
        RI2["<b>Update OpenSearch</b><br/>Blue-green swap"]:::store
        RI3["<b>Update ranking model</b><br/>Si nueva versión MLflow"]:::tool
        RI4["<b>Warm-up cache</b><br/>Queries más frecuentes"]:::serve
    end

    IX3 & RK4 ==> RI1
    RI1 ==> RI2
    RI1 ==> RI3
    RI2 & RI3 ==> RI4

    %% ═══════════════════════════════════════════════════════════
    %% 6 · QUERY-TIME FLOW
    %% ═══════════════════════════════════════════════════════════
    subgraph SERVING["<b>6 · QUERY-TIME FLOW</b><br/>&lt; 100 ms end-to-end"]
        QT1["<b>API Gateway</b><br/>User query"]:::serve
        QT2["<b>Search Service</b><br/>EKS Pod"]:::serve
        QT3["<b>Query Preprocessing</b><br/>Spell correction SymSpell<br/>Sinónimos · Tokenización"]:::query
        QT4["<b>BM25 Lexical</b><br/>OpenSearch text → top 100"]:::store
        QT5["<b>KNN Semantic</b><br/>Query embedding → top 100"]:::store
        QT6["<b>Merge + Dedup</b><br/>~150 candidates"]:::data
        QT7["<b>LightGBM Predict</b><br/>Score 150 → top K = 20"]:::model
        QT8["<b>Post-processing</b><br/>Stock &gt; 0 · No reportados<br/>Sponsored boost<br/>Seller diversity max 3"]:::query
        QT9["<b>Response</b><br/>API Gateway → Frontend<br/>&lt; 100 ms p95"]:::serve
    end

    SD -.-> QT3
    QT1 ==> QT2
    QT2 ==> QT3
    QT3 ==> QT4 & QT5
    QT4 & QT5 ==> QT6
    QT6 ==> QT7
    QT7 ==> QT8
    QT8 ==> QT9
    RI2 -.-> QT4
    RI2 -.-> QT5
    RI4 -.-> QT3

    %% ── Subgraph styling ──
    style FUENTES2 fill:#fef3c7,stroke:#f59e0b,rx:10,color:#78350f
    style DATAPREP fill:#dbeafe,stroke:#3b82f6,rx:10,color:#1e3a5f
    style EMBEDDINGS fill:#ede9fe,stroke:#8b5cf6,rx:10,color:#3b0764
    style INDEX fill:#ccfbf1,stroke:#14b8a6,rx:10,color:#134e4a
    style RANKING fill:#fce7f3,stroke:#ec4899,rx:10,color:#831843
    style REINDEX fill:#fff7ed,stroke:#f97316,rx:10,color:#7c2d12
    style SERVING fill:#d1fae5,stroke:#10b981,rx:10,color:#064e3b
```

### 5.3 Algoritmos y Justificación

El buscador funciona en 3 etapas: primero busca por palabras y por significado en paralelo, luego combina y ordena los resultados, y finalmente los entrega en menos de 100 milisegundos.

| Etapa | Algoritmo | Qué hace (en palabras simples) | Por qué este algoritmo |
|---|---|---|---|
| **Embeddings** | **Sentence-BERT** (modelo pre-entrenado y ajustado a DSRPMart) | Convierte títulos de productos y búsquedas del usuario a vectores numéricos que capturan el significado | Equilibrio óptimo entre velocidad y calidad. Entiende que "celular" ≈ "smartphone" |
| **Búsqueda por palabras** | **BM25** (integrado en OpenSearch) | Busca coincidencias de texto tradicional, like Google | Captura matches exactos que la IA podría perder (ej: "iPhone 15 Pro Max 256GB") |
| **Búsqueda por significado** | **KNN HNSW** (Amazon OpenSearch) | Busca los productos cuyo significado es más cercano a lo que el usuario escribió | Búsqueda rápida entre millones de vectores en menos de 20ms |
| **Ranking** | **LightGBM LambdaRank** | Combina señales de palabras + significado + popularidad + historial del usuario en un solo puntaje | Optimiza directamente la calidad del orden de resultados |
| **Corrección** | **SymSpell** | Corrige errores de escritura del usuario ("iphne" → "iphone") | Ultra-rápido sin afectar la latencia |
| **Sinónimos** | **Diccionario curado + Word2Vec** | Expande "celular" → "smartphone", "teléfono móvil" | Reduce drásticamente las búsquedas sin resultado |

**Optimizaciones** (mejoran velocidad y relevancia):

- **Búsqueda híbrida** — Combina búsqueda por palabras y por significado en paralelo, encontrando más resultados relevantes que usando solo una de las dos
- **Compresión de embeddings** — Reduce el tamaño del índice de búsqueda 4 veces, haciendo las consultas 2 veces más rápidas
- **Cache de resultados** — Las búsquedas frecuentes se guardan en memoria (Redis) por 30 minutos para responder instantáneamente
- **Pre-calentamiento** — Las 1.000 búsquedas más populares se pre-calculan al desplegar un nuevo índice

### 5.4 Model Card – Motor de Búsqueda

> Ficha técnica del modelo de búsqueda inteligente, siguiendo el mismo formato de documentación del recomendador.

> **MODEL CARD — DSRPMart Search Engine v1.0**

**Información General**

| Atributo | Detalle |
|---|---|
| **Nombre del modelo** | `search_engine` (embedding + ranker) |
| **Versión** | 1.0.0 |
| **Tipo** | Sentence-BERT (Retrieval) + LambdaRank (Ranking) |
| **Framework** | Sentence-Transformers 2.x + LightGBM 4.3 |
| **Propietario** | Equipo Data Science – DSRPMart |
| **Fecha creación** | Febrero 2026 |
| **Frecuencia retrain** | Semanal (embeddings) + Diario (ranker) |

**Datos de Entrenamiento**

| Atributo | Detalle |
|---|---|
| **Período** | Últimos 6 meses de búsquedas |
| **Volumen** | ~50M pares (query, product_clicked) / ~200K queries únicas / ~500K productos |
| **Fuente** | S3 `s3://dsrpmart-data/processed/search/` |
| **Split** | Temporal – Train 80% / Val 10% / Test 10% |

**Métricas de Evaluación (Offline)**

| Métrica | Valor |
|---|---|
| NDCG@10 | 0.452 |
| MRR | 0.387 |
| Recall@100 (retrieval) | 0.91 |
| Precision@5 | 0.34 |
| Zero-result rate | < 2% de queries |

**Métricas de Negocio Impactadas**

- Search CTR (clicks en resultados / búsquedas)
- Search Conversion Rate (compra tras búsqueda)
- Zero-result Rate (búsquedas sin resultados)
- Search Exit Rate (abandono tras búsqueda)
- Revenue per Search (ingresos atribuidos a búsqueda)

**Limitaciones**

- Nuevos productos (< 24h) solo tienen retrieval lexical hasta re-index
- Queries muy largas (> 20 tokens) se truncan
- Idioma: solo español (Latam). No soporta queries en otros idiomas
- Ranking personalizado solo para usuarios logueados con > 5 eventos

**Umbrales de Alerta**

| Umbral | Acción |
|---|---|
| Latencia p95 > 100ms | Escalar pods Search Service |
| Zero-result rate > 5% | Revisar índice y sinónimos |
| Search CTR < 0.25 | Análisis de relevancia + retraining |
| Embedding drift > 0.20 | Re-fine-tune Sentence-BERT |

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

> **¿Por qué importa esta decisión?** Elegir entre procesar datos por lotes (batch), en tiempo real o en streaming afecta directamente la experiencia de compra y los costos. La pregunta clave no es "qué es más moderno" sino "qué maximiza las ventas al menor costo". A continuación analizamos cada caso desde lo que el usuario espera al usar la app.

### 6.1 Productos Recomendados → **Batch (Mini-Batch Frecuente)**

| Criterio | Análisis | Decisión |
|---|---|---|
| Frecuencia requerida | "Varias veces al día" → No necesita real-time sub-segundo | **Batch cada 6h (4 veces al día)** |
| Volumen | ~10M usuarios × TOP-20 = 200M predicciones | Spark batch es más eficiente que real-time |
| Latencia aceptable | Rankings pre-calculados servidos desde Redis (< 5ms) | Batch + cache cumple SLA |
| Costo | Spot instances GPU para training, Spot CPU para inference → ~70% ahorro vs. on-demand | Batch optimiza costos |
| Complejidad del modelo | Two-Tower + LambdaRank son pesados para real-time per-request | Batch permite modelos más complejos |

**¿Por qué NO en tiempo real?**

- Las recomendaciones no necesitan actualizarse al instante; cada 6 horas es suficiente para una buena experiencia
- Calcular recomendaciones en tiempo real por cada visita requiere servidores GPU costosos
- El enfoque batch + cache da tiempos de respuesta de 5 milisegundos a una fracción del costo

**¿Por qué NO streaming?**

- No hay eventos individuales que invaliden todo el ranking (si un producto se agota, se filtra en el momento de servirlo)
- La complejidad adicional de streaming no justifica la pequeña mejora en frescura de las recomendaciones

### 6.2 Motor de Búsqueda → **Hybrid: Batch (Indexación) + Real-Time (Serving)**

| Criterio | Análisis | Decisión |
|---|---|---|
| Indexación de productos | Embeddings y ranking model se actualizan diario | **Batch** (Airflow scheduled) |
| Serving de queries | El usuario espera resultados inmediatos al escribir | **Real-time** (< 100ms p95) |
| SLA de latencia | UX de búsqueda requiere sub-segundo | Serving en EKS + Redis cache |
| Stock freshness | Productos sin stock no deben aparecer | CDC + filtro en query-time |

El Motor de Búsqueda es naturalmente **híbrido**: la preparación de índices se ejecuta por lotes (batch) pero las respuestas al usuario son en tiempo real. El modelo de ranking es lo suficientemente ligero como para evaluar 150 productos en menos de 5 milisegundos por cada búsqueda.

**Impacto en ventas:** Una búsqueda que responde en menos de 100 milisegundos y muestra resultados relevantes reduce el abandono y aumenta las compras. El 8% de búsquedas sin resultado hoy representa ventas perdidas que esta arquitectura busca recuperar.

---

## 7. Stack Tecnológico AWS Cloud-Native y Justificación Comparativa

> **Criterio de selección:** Cada herramienta fue elegida por su capacidad de **mejorar las ventas** de DSRPMart (más conversión, búsqueda más rápida, recomendaciones más frescas), no solo por méritos técnicos. Usamos **AWS como proveedor único** para simplificar la operación y aprovechar el ecosistema de datos más maduro del mercado, con la mayor base de profesionales certificados en Latinoamérica.

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
| **MLflow Tracking Server** (desplegado en EKS) | Registro de experimentos: parámetros, métricas, archivos generados | Código abierto, funciona con cualquier framework, interfaz web |
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

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD
    subgraph CI["CI — GitHub Actions  ·  on PR + merge to main"]
        direction TB
        LINT["<b>Job 1: Lint & Test</b><br>ruff lint · mypy · pytest --cov >80%<br>integration tests (mock) · Great Expectations"]:::ci
        BUILD["<b>Job 2: Build & Push</b><br>Docker build linux/amd64 → ECR :sha + :latest<br>Trivy vulnerability scan · cosign sign"]:::ci
        MLTEST["<b>Job 3: ML Pipeline Test</b><br>DVC repro --dry · Trigger KFP staging<br>Compare metrics vs Champion (MLflow API)<br>Auto-comment PR with comparison table"]:::ci
        LINT --> BUILD --> MLTEST
    end

    subgraph CD["CD — ArgoCD GitOps  ·  Continuous Deployment"]
        direction TB
        HELM["<b>Step 1: Update Helm Values</b><br>image.tag → values-prod.yaml<br>Commit dsrpmart-infra · PR approval"]:::cd
        SYNC["<b>Step 2: ArgoCD Sync</b><br>Reconcile EKS cluster state<br>Rolling update pods · zero-downtime<br>readinessProbe + livenessProbe"]:::cd
        POST["<b>Step 3: Post-Deploy</b><br>Smoke tests (e2e staging)<br>MLflow: Staging → Production<br>Slack #ml-deployments"]:::cd
        HELM --> SYNC --> POST
    end

    subgraph RB["ROLLBACK PROCEDURE"]
        direction LR
        AUTO["<b>Automático</b><br>ArgoCD: CrashLoop → reverts"]:::rb
        MANUAL["<b>Manual</b><br>git revert → ArgoCD reconcilia"]:::rb
        MLRB["<b>MLflow</b><br>Promote versión anterior"]:::rb
        PD["<b>PagerDuty</b><br>On-call MLOps alert"]:::rb
    end

    MLTEST ==>|"✅ All checks pass"| HELM
    POST -.->|"❌ Health check fail"| AUTO

    classDef ci fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:2px
    classDef cd fill:#15803d,stroke:#166534,color:#fff,stroke-width:2px
    classDef rb fill:#b91c1c,stroke:#991b1b,color:#fff,stroke-width:1px

    style CI fill:#eef2ff,stroke:#818cf8,stroke-width:2px,rx:10
    style CD fill:#dcfce7,stroke:#22c55e,stroke-width:2px,rx:10
    style RB fill:#fef2f2,stroke:#fca5a5,stroke-width:1px,rx:10
```

### 7.g Métricas de Performance, Aplicación y Herramientas de Visualización

| Tipo de Métrica | Métricas | Herramienta que Recolecta | Herramienta de Visualización |
|---|---|---|---|
| **Rendimiento del modelo (offline)** | NDCG@10, MRR, Hit Rate, Recall@100 | MLflow Tracking | MLflow UI + Grafana |
| **Rendimiento del modelo (online)** | CTR, Tasa de Conversión, Ingreso por Sesión | Kinesis → Redshift | Amazon QuickSight + Grafana |
| **Cambio en datos (Data Drift)** | PSI por variable, KS test, cambio en distribución de predicciones | Evidently AI (reportes batch) | Evidently Dashboard + reportes HTML en S3 |
| **Infraestructura K8s** | CPU/Memoria por pod, reinicios, escalamiento de nodos | Prometheus (kube-state-metrics) | Grafana dashboards |
| **Aplicación / API** | Latencia p50/p95/p99, tasa de error, consultas por segundo | Prometheus (métricas FastAPI) | Grafana |
| **Salud de Pipelines** | Tasa de éxito de DAGs, duración de tareas, incumplimiento de SLA | Métricas Airflow + CloudWatch | Grafana + Airflow UI |
| **Negocio** | Ventas totales (GMV), Incremento de ingresos, Conversión, Retención | Agregaciones en Redshift | Amazon QuickSight |
| **Costos** | Gasto por servicio, costo por predicción, costo por experimento | AWS Cost Explorer + tags personalizados | QuickSight + Grafana |

**Stack de Observabilidad:**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart LR
    subgraph COLLECT["DATA COLLECTION"]
        direction TB
        PROM["<b>Prometheus</b><br>kube-state-metrics<br>node-exporter<br>custom app metrics"]:::prom
        CW["<b>CloudWatch</b><br>MWAA Airflow · ElastiCache<br>OpenSearch · Kinesis"]:::aws
        EV["<b>Evidently AI</b><br>Feature drift · Prediction drift<br>Target drift (batch reports)"]:::ev
    end

    subgraph VIZ["VISUALIZATION"]
        direction TB
        GF["<b>Grafana Dashboards</b><br>ML Model Health<br>Search Latency · Rec KPIs<br>Infrastructure · Pipeline Health"]:::grafana
    end

    subgraph ALERT["ALERTING"]
        direction TB
        AM["<b>AlertManager</b>"]:::alert
        PD["PagerDuty<br>on-call"]:::alert
        SL["Slack<br>#ml-alerts"]:::alert
        AM --> PD & SL
    end

    PROM & CW & EV ==> GF
    PROM --> AM

    classDef prom fill:#e6522c,stroke:#c43e1f,color:#fff,stroke-width:2px
    classDef aws fill:#232f3e,stroke:#ff9900,color:#ff9900,stroke-width:2px
    classDef ev fill:#7c3aed,stroke:#6d28d9,color:#fff,stroke-width:1px
    classDef grafana fill:#f46800,stroke:#d45800,color:#fff,stroke-width:2px
    classDef alert fill:#dc2626,stroke:#b91c1c,color:#fff,stroke-width:1px

    style COLLECT fill:#f8fafc,stroke:#cbd5e0,stroke-width:1px,rx:10
    style VIZ fill:#fff7ed,stroke:#fdba74,stroke-width:2px,rx:10
    style ALERT fill:#fef2f2,stroke:#fca5a5,stroke-width:1px,rx:10
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

> Esta sección justifica cada elección tecnológica comparando alternativas reales del mercado. Para cada decisión se evalúan criterios con peso y se asigna una puntuación de 1 a 5 estrellas. Esto permite explicar con transparencia **por qué elegimos cada herramienta y por qué descartamos las demás**.

#### 7.i.1 Proveedor de Nube: ¿Por qué AWS y no GCP o Azure?

| Criterio (peso) | AWS | GCP | Azure | Justificación de elección |
|---|:---:|:---:|:---:|---|
| **Servicios de ML managed** (25%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | SageMaker, Bedrock, OpenSearch nativo; Vertex AI de GCP es comparable pero AWS tiene mayor adopción enterprise |
| **Kubernetes managed (EKS)** (20%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GKE es ligeramente superior en UX, pero EKS + Karpenter cierra la brecha. AWS IRSA es mejor que GCP Workload Identity en flexibilidad |
| **Ecosistema de datos** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Kinesis + Redshift + S3 + Glue + DMS = ecosistema más maduro para data pipelines |
| **Talento disponible** (15%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | AWS tiene la mayor base de profesionales certificados; más fácil reclutar |
| **Costos Spot/Preemptible** (10%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | GCP Preemptible es más barato, pero AWS Spot tiene mejor disponibilidad y el ahorro total es similar |
| **Multi-region/DR** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | AWS tiene más regiones globales, importante para expansión de DSRPMart en Latam |
| **Score ponderado** | **4.55** | **4.20** | **3.80** | **→ AWS elegido** |

**Decisión:** AWS gana por su ecosistema de datos maduro, la cantidad de profesionales certificados disponibles en Latam y su mayor presencia de data centers en la región. GCP sería la segunda opción por su fortaleza en Kubernetes y precios competitivos.

#### 7.i.2 Orquestación: ¿Por qué Airflow (MWAA) + Kubeflow y no Prefect, Dagster o solo SageMaker Pipelines?

| Criterio (peso) | Airflow (MWAA) + KFP | Prefect 2.0 | Dagster | SageMaker Pipelines |
|---|:---:|:---:|:---:|:---:|
| **Madurez y comunidad** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **AWS managed service** (20%) | ⭐⭐⭐⭐⭐ (MWAA) | ⭐⭐ (self-hosted) | ⭐⭐ (self-hosted) | ⭐⭐⭐⭐⭐ |
| **ML-specific features** (20%) | ⭐⭐⭐⭐⭐ (KFP) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU scheduling en K8s** (15%) | ⭐⭐⭐⭐⭐ (KFP nativo) | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ (propio infra) |
| **Vendor lock-in** (dependencia del proveedor) (15%) | ⭐⭐⭐⭐ (open-source) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ (SageMaker lock-in) |
| **DAGs como código Python** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Score ponderado** | **4.60** | **3.15** | **3.35** | **3.50** |

**Decisión:** La combinación **Airflow como orquestador general + Kubeflow para las tareas específicas de ML** es única en que: (1) MWAA es un servicio administrado por AWS, eliminando el trabajo de mantener Airflow, (2) Kubeflow permite ejecutar cada paso del entrenamiento en contenedores independientes con acceso a GPU, (3) ambos son de código abierto, evitando depender de un proveedor único. SageMaker Pipelines se descartó porque genera dependencia total del ecosistema propietario de AWS.

#### 7.i.3 Model Management: ¿Por qué MLflow y no Weights & Biases, Neptune AI o SageMaker Model Registry?

| Criterio (peso) | MLflow (self-hosted EKS) | W&B (SaaS) | Neptune AI (SaaS) | SageMaker Registry |
|---|:---:|:---:|:---:|:---:|
| **Costo** (25%) | ⭐⭐⭐⭐⭐ (gratis, OSS) | ⭐⭐ ($$$) | ⭐⭐⭐ ($$) | ⭐⭐⭐⭐ (incluido) |
| **Model Registry integrado** (20%) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Compatible con cualquier framework** (20%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (sesgo a SageMaker) |
| **API programable** (15%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Control de ubicación de datos** (10%) | ⭐⭐⭐⭐⭐ (auto-hospedado) | ⭐⭐ (SaaS en EEUU) | ⭐⭐ (SaaS) | ⭐⭐⭐⭐ |
| **Comunidad y plugins** (10%) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Score ponderado** | **4.65** | **3.50** | **3.30** | **3.50** |

**Decisión:** MLflow auto-hospedado en EKS es la mejor opción para una startup porque: (1) **sin costo de licencia**, solo se paga la infraestructura (base de datos + almacenamiento), (2) **control total** de los datos dentro de la red privada de AWS, (3) **API completa** para automatizar todo el proceso de CI/CD, (4) **funciona con cualquier framework** — TensorFlow, LightGBM, Sentence-Transformers, etc. W&B es excelente pero su costo (~$50/usuario/mes) no se justifica en esta fase.

#### 7.i.4 Feature Store: ¿Por qué Feast y no Tecton o SageMaker Feature Store?

| Criterio | Feast (OSS) | Tecton | SageMaker Feature Store |
|---|:---:|:---:|:---:|
| **Costo** | ⭐⭐⭐⭐⭐ (gratis) | ⭐ ($$$$$) | ⭐⭐⭐ ($$) |
| **Point-in-time joins** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Online store: Redis** | ⭐⭐⭐⭐⭐ (nativo) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (DynamoDB) |
| **Offline store: S3/Parquet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Dependencia del proveedor** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Complejidad operacional** | ⭐⭐⭐ (self-managed) | ⭐⭐⭐⭐⭐ (SaaS) | ⭐⭐⭐⭐ |

**Decisión:** Feast es de código abierto, se despliega en el mismo clúster de Kubernetes, usa el Redis que ya tenemos para servir recomendaciones, y S3 para almacenamiento histórico. No agrega costo significativo. Tecton es técnicamente superior pero su precio (más de $100.000/año) no tiene sentido para una startup.

#### 7.i.5 CI/CD: ¿Por qué GitHub Actions + ArgoCD y no Jenkins, GitLab CI o AWS CodePipeline?

| Criterio | GitHub Actions + ArgoCD | Jenkins | GitLab CI | AWS CodePipeline |
|---|:---:|:---:|:---:|:---:|
| **Integración con código** | ⭐⭐⭐⭐⭐ (GitHub nativo) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (GitLab) | ⭐⭐⭐ |
| **GitOps para K8s** | ⭐⭐⭐⭐⭐ (ArgoCD) | ⭐⭐ (plugin) | ⭐⭐⭐ | ⭐⭐ |
| **Costo operacional** | ⭐⭐⭐⭐ (SaaS + OSS) | ⭐⭐ (self-hosted) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **ML pipeline integration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Ecosistema marketplace** | ⭐⭐⭐⭐⭐ (Actions Market) | ⭐⭐⭐⭐ (plugins) | ⭐⭐⭐ | ⭐⭐ |
| **Auto-rollback K8s** | ⭐⭐⭐⭐⭐ (ArgoCD native) | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**Decisión:** GitHub Actions se encarga de las pruebas y construcción de imágenes (CI) y ArgoCD se encarga del despliegue a Kubernetes (CD). Esta separación sigue las mejores prácticas de la industria. ArgoCD además detecta cuando el estado del clúster difiere de lo definido en Git y lo corrige automáticamente, algo que Jenkins o CodePipeline no hacen de forma nativa. Jenkins se descartó por el alto costo de mantener un servidor dedicado.

#### 7.i.6 IaC: ¿Por qué Terraform y no AWS CDK, Pulumi o CloudFormation?

| Criterio | Terraform | AWS CDK | Pulumi | CloudFormation |
|---|:---:|:---:|:---:|:---:|
| **Multi-cloud portable** | ⭐⭐⭐⭐⭐ | ⭐ (AWS only) | ⭐⭐⭐⭐ | ⭐ (AWS only) |
| **Ecosistema providers** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **State management** | ⭐⭐⭐⭐ (S3 backend) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Comunidad y módulos** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Curva de aprendizaje** | ⭐⭐⭐⭐ (HCL) | ⭐⭐⭐⭐⭐ (TypeScript) | ⭐⭐⭐⭐ (Python) | ⭐⭐ (YAML/JSON) |
| **Plan/Preview** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (cdk diff) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (change set) |

**Decisión:** Terraform por su portabilidad (si DSRPMart decide usar más de un proveedor de nube en el futuro), la madurez de su ecosistema, y la claridad del comando `terraform plan` que permite revisar los cambios antes de aplicarlos en infraestructura real. AWS CDK sería segunda opción para equipos que prefieren programar en TypeScript.

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

> **¿Por qué esta estrategia?** Un modelo de ML solo genera valor cuando los usuarios interactúan con él en producción. Pero si desplegamos un modelo nuevo que funciona peor que el actual, perdemos ventas. Por eso usamos una estrategia gradual: el modelo actual ("champion") sigue atendiendo a los usuarios mientras el modelo nuevo ("challenger") se prueba progresivamente. Solo cuando el challenger **demuestra con datos** que mejora las ventas, se promueve al 100%.

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

1. Ambos procesan datos por lotes, lo que permite probar el modelo nuevo sin afectar al usuario
2. Ambos impactan métricas de negocio que podemos medir directamente (clicks, conversión, ingresos)
3. La infraestructura compartida (EKS + Redis + MLflow) soporta múltiples versiones de forma nativa
4. Los A/B tests requieren un volumen mínimo de tráfico para ser estadísticamente válidos, y DSRPMart tiene ese volumen

### 8.2 Diagrama de Proceso de Despliegue

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD
    subgraph F0["<b>FASE 0 · DESARROLLO + EXPERIMENTACIÓN</b>"]
        direction TB
        DEV_DS["<b>Data Scientist en branch feature/*</b><br>JupyterHub · Kubeflow Notebooks"]:::phase0
        DEV_EXP["<b>Experimentación</b><br>MLflow experiment tracking<br>Feast offline store · DVC dataset versioning"]:::phase0
        DEV_PR["<b>Métricas offline prometedoras</b><br>→ PR a develop"]:::phase0
        DEV_DS ==> DEV_EXP ==> DEV_PR
    end

    DEV_PR ==>|"PR + Code Review"| F1_MERGE

    subgraph F1["<b>FASE 1 · VALIDACIÓN EN STAGING</b>"]
        direction TB
        F1_MERGE["<b>PR merged → GitHub Actions CI/CD</b><br>Tests: unit + integration + data quality"]:::phase1
        F1_BUILD["<b>Docker image built → ECR :sha</b><br>KFP Pipeline triggered en staging EKS"]:::phase1
        F1_EVAL["<b>Evaluación Staging</b><br>Train con datos staging · Métricas vs Champion<br>Evidently drift report"]:::phase1
        F1_REG["<b>IF pass → MLflow stage=Staging</b><br>Slack: Model v X ready for shadow"]:::phase1
        F1_MERGE ==> F1_BUILD ==> F1_EVAL ==> F1_REG
    end

    F1_REG ==>|"NDCG pass + drift OK"| F2_SHADOW

    subgraph F2["<b>FASE 2 · SHADOW MODE — 1 semana</b>"]
        direction TB
        F2_SHADOW["<b>Challenger predicciones en PARALELO</b><br>No se muestran al usuario"]:::phase2
        F2_REC["<b>Recomendador:</b> batch inference →<br>Redis ns challenger · solo almacenadas"]:::phase2
        F2_SEARCH["<b>Buscador:</b> ranking challenger vs champion<br>logged only · usuario ve Champion"]:::phase2
        F2_COMP["<b>Comparación diaria automática</b><br>NDCG · MRR · Coverage<br>Evidently drift report"]:::phase2
        F2_DASH["<b>Dashboard Grafana</b><br>Champion vs Challenger side-by-side"]:::phase2
        F2_SHADOW ==> F2_REC & F2_SEARCH
        F2_REC & F2_SEARCH ==> F2_COMP ==> F2_DASH
    end

    F2_DASH ==>|"Métricas estables"| F3_SPLIT

    subgraph F3["<b>FASE 3 · A/B TEST — 1-2 semanas</b>"]
        direction TB
        F3_SPLIT["<b>Split de tráfico</b><br>80% Champion / 20% Challenger"]:::phase3
        F3_REC["<b>Recomendador</b><br>Redis: user:id:recs champion<br>Redis: user:id:recs:challenger test<br>Frontend routing user_id % 100"]:::phase3
        F3_SEARCH2["<b>Buscador</b><br>model_version param · user_id hash<br>sticky sessions"]:::phase3
        F3_KPI["<b>Medición KPIs por grupo</b><br>CTR · Conversion · Revenue · Session Time<br>t-test p < 0.05 · MDE 2% CTR lift"]:::phase3
        F3_QUICK["<b>Dashboard A/B</b><br>Redshift → QuickSight"]:::phase3
        F3_SPLIT ==> F3_REC & F3_SEARCH2
        F3_REC & F3_SEARCH2 ==> F3_KPI ==> F3_QUICK
    end

    F3_QUICK ==> GATE

    GATE{"<b>¿Challenger supera Champion?</b><br>KPI lift > threshold · p < 0.05"}:::gate

    GATE ==>|"SÍ · Significativo"| PROMOTE
    GATE ==>|"NO"| ROLLBACK

    subgraph F4A["<b>FASE 4a · PROMOCIÓN</b>"]
        direction TB
        PROMOTE["<b>MLflow: Challenger → Production</b><br>Champion → Archived"]:::success
        ARGO["<b>ArgoCD: update ConfigMap</b><br>model_version → nueva versión"]:::success
        ROLLOUT["<b>Full rollout 100%</b>"]:::success
        MON48["<b>Monitoring intensivo 48h</b><br>Prometheus · Grafana · PagerDuty"]:::monitor
        PROMOTE ==> ARGO ==> ROLLOUT ==> MON48
    end

    subgraph F4B["<b>FASE 4b · ROLLBACK</b>"]
        direction TB
        ROLLBACK["<b>Challenger archivado</b><br>Champion continúa operando"]:::fail
        POSTMORTEM["<b>Post-mortem doc</b><br>Learnings → backlog"]:::fail
        NEWEXP["Nuevos experimentos"]:::neutral
        ROLLBACK ==> POSTMORTEM ==> NEWEXP
    end

    F2_DASH -.->|"Grafana"| GRAFANA_MON["📊 Grafana Dashboards"]:::monitor
    F3_QUICK -.->|"QuickSight"| GRAFANA_MON
    MON48 -.-> GRAFANA_MON

    classDef phase0 fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px
    classDef phase1 fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:2px
    classDef phase2 fill:#7c3aed,stroke:#6d28d9,color:#fff,stroke-width:2px
    classDef phase3 fill:#ca8a04,stroke:#a16207,color:#fff,stroke-width:2px
    classDef gate fill:#fef3c7,stroke:#d97706,color:#92400e,stroke-width:3px,font-weight:bold
    classDef success fill:#15803d,stroke:#166534,color:#fff,stroke-width:3px,font-weight:bold
    classDef fail fill:#b91c1c,stroke:#991b1b,color:#fff,stroke-width:2px
    classDef monitor fill:#0e7490,stroke:#155e75,color:#fff,stroke-width:1px
    classDef neutral fill:#f1f5f9,stroke:#94a3b8,color:#475569,stroke-width:1px

    style F0 fill:#f0f4ff,stroke:#1e3a5f,stroke-width:2px,rx:8
    style F1 fill:#eff6ff,stroke:#2563eb,stroke-width:2px,rx:8
    style F2 fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px,rx:8
    style F3 fill:#fefce8,stroke:#ca8a04,stroke-width:2px,rx:8
    style F4A fill:#f0fdf4,stroke:#15803d,stroke-width:2px,rx:8
    style F4B fill:#fef2f2,stroke:#b91c1c,stroke-width:2px,rx:8
```

---

## 9. Pasos de Construcción, Actores y Colaboración

> **Plan de trabajo:** Construimos cada caso de uso en aproximadamente 16 semanas (4 meses), empezando con una semana de descubrimiento donde definimos con el área de producto qué métricas de ventas queremos mejorar. Al final del proceso, un A/B test compara el modelo nuevo contra el sistema actual para verificar que realmente mejora los resultados antes de activarlo para todos los usuarios.

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

**Organización del Equipo DSRPMart ML**

| Rol | Personas | Responsabilidades |
|---|:---:|---|
| **ML Lead / Arquitecto** | 1 | Diseño de arquitectura MLOps, Model Cards, decisiones de stack, go/no-go de despliegues, revisión técnica de modelos y pipelines |
| **Data Scientist Sr** | 1 | Two-Tower, Sentence-BERT: modelos complejos, research de arquitecturas, tuning avanzado |
| **Data Scientist** | 2 | LambdaRank, Item2Vec, feature engineering, EDA, experimentación, evaluación offline |
| **MLOps Engineer** | 2 | Kubeflow Pipelines, Airflow DAGs, CI/CD, batch inference, monitoring, Evidently, MLflow administration, on-call rotación |
| **Data Engineer** | 1 | Kinesis ingesta, Spark jobs, DMS CDC, Redshift DWH, data quality contracts |
| **ML Platform Engineer** | 1 | EKS cluster, Terraform, ArgoCD, Feast, OpenSearch, ElastiCache, seguridad (IRSA) |
| **Backend Engineer** | 1 | FastAPI Search Service, API Gateway, frontend integration, Redis client optimization |
| **Product Owner** | 1 | Definir KPIs, priorizar features, stakeholder management, A/B test analysis review |
| **QA / Analytics** | 1 | A/B test statistical analysis, data validation, regression testing, UAT |
| **TOTAL** | **11** | |

### 9.4 Modelo de Colaboración

**Proceso de Trabajo — Scrum Adaptado para Machine Learning**

**Cadencia:**

| Ceremonia | Frecuencia |
|---|---|
| Sprint | 2 semanas |
| Daily Standup | 15 min (async en Slack los viernes) |
| Sprint Planning | Lunes S1 (2h) |
| Sprint Review | Viernes S2 (1h) — demo de métricas + pipeline |
| Retro | Viernes S2 (30 min) |
| ML Review | Miércoles S2 (1h) — revisión técnica de modelos |

**Artefactos:**

- **Product Backlog:** Jira Board "DSRPMart ML"
- **Sprint Backlog:** Jira + GitHub Projects
- **RFC (Request For Comments):** Documento técnico en Notion (requerido para: nuevo modelo, cambio de stack, nuevo pipeline)
- **Model Card:** Actualizado en cada release (en repo Git)
- **Runbook:** Procedimientos de on-call y rollback en Confluence
- **ADR (Architecture Decision Records):** En repo dsrpmart-infra/

**Colaboración entre equipos:**

| Interacción | Mecanismo |
|---|---|
| DS ↔ MLOps | El Data Scientist entrega su código reproducible y el ID del experimento en MLflow → MLOps lo convierte en un pipeline automatizado con CI/CD |
| DE ↔ ML | Los equipos comparten un contrato de datos formal: Data Engineering publica las features en Feast y ML las consume de forma reproducible |
| Backend ↔ ML | Contrato de API documentado (formato OpenAPI) con acuerdos de rendimiento: latencia máxima, capacidad mínima, margen de error permitido |
| Product ↔ ML | Antes de cada modelo se define un documento de KPIs. Después del A/B test hay una reunión de Go/No-Go para decidir si se promueve el modelo |

---

## 10. Diagramas de Arquitectura y Flujos de Proceso

> Los siguientes 3 diagramas muestran cómo se conectan todos los componentes del sistema: desde que los datos entran hasta que el usuario ve las recomendaciones o resultados de búsqueda. Cada diagrama cubre una perspectiva diferente: el flujo de entrenamiento del modelo, la arquitectura completa en AWS, y el proceso de CI/CD (integración y despliegue continuo).

### 10.a End-to-End Entrenamiento de Modelo (Ambos Modelos)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD

    %% ── MWAA Scheduler ──────────────────────────────────────────────
    subgraph MWAA["☁️ Amazon MWAA — Managed Airflow"]
        direction LR
        SCHED["<b>DAG Scheduler</b><br/>product_recommender_train: daily 00:00 UTC<br/>search_engine_train: daily 02:00 UTC<br/><i>(Batch inference: 4×/día en DAG separado)</i>"]
    end

    %% ── Data Ingestion & Validation ─────────────────────────────────
    subgraph INGEST["📥 Data Ingestion & Validation"]
        direction LR
        S3SENSOR["<b>S3 Sensor</b><br/>Detect new data<br/><i>s3://dsrpmart-data/raw/</i>"]
        VALIDATE["<b>Data Validation</b><br/>Great Expectations<br/>Schema + Nulls + Quality<br/><i>→ s3://…/validated/</i>"]
        S3SENSOR ==> VALIDATE
    end

    %% ── Feature Engineering ─────────────────────────────────────────
    subgraph FEAT["⚙️ Feature Engineering"]
        direction LR
        SPARK_FE["<b>Spark on EKS</b><br/>Transform & compute<br/>features at scale"]
        FEAST["<b>Feast Feature Store</b><br/>Online + Offline store<br/><i>→ s3://…/features/</i>"]
        SPARK_FE ==> FEAST
    end

    %% ── KFP Training Pipeline ───────────────────────────────────────
    subgraph KFP["🔬 Kubeflow Pipeline — EKS Cluster"]
        direction LR
        TRAIN["<b>Train Model</b><br/>GPU Pod · p3.2xlarge<br/>MLflow log params,<br/>metrics & artifacts"]
        EVAL["<b>Evaluate</b><br/>CPU Pod<br/>vs Champion model<br/>NDCG · MRR · Evidently<br/>PSI · KS drift test"]
        REGISTER["<b>Register Model</b><br/>CPU Pod<br/>MLflow Registry<br/>stage = Staging<br/>tags: git_sha,<br/>dataset_ver, date"]
        TRAIN ==> EVAL ==> REGISTER
    end

    %% ── Decision Gate ───────────────────────────────────────────────
    GATE{"<b>Quality Gate</b><br/>PASS / FAIL"}
    FAIL_NODE["<b>FAIL</b><br/>SNS Alert → Slack<br/>Abort DAG"]
    PASS_NODE["<b>PASS</b><br/>SNS → Slack<br/>Continue"]

    %% ── Batch Inference ─────────────────────────────────────────────
    subgraph BATCH["🚀 Batch Inference — Spark on EKS"]
        direction LR
        INFER["<b>Batch Predict</b><br/>Load model: MLflow Production<br/>Load features: Feast"]
        REDIS["<b>Redis</b><br/>TOP-20 recs/user<br/>ElastiCache TTL"]
        S3OUT["<b>S3 + Redshift</b><br/>Predictions audit log"]
        OSEARCH["<b>OpenSearch</b><br/>Product embeddings<br/>search index"]
        INFER ==> REDIS
        INFER ==> S3OUT
        INFER ==> OSEARCH
    end

    %% ── Artifact Storage ────────────────────────────────────────────
    subgraph STORE["🗄️ Artifact Storage"]
        direction LR
        S3BUCK["<b>S3 Buckets</b><br/>dsrpmart-data/raw<br/>dsrpmart-data/validated<br/>dsrpmart-data/processed<br/>dsrpmart-features<br/>dsrpmart-models<br/>dsrpmart-mlflow-artifacts"]
        MLFLOW["<b>MLflow Tracking Server</b><br/>Experiments · Runs<br/>Params · Metrics<br/>Artifacts (S3-backed)<br/>Model Registry<br/>Staging → Production"]
        S3BUCK ~~~ MLFLOW
    end

    %% ── Main Flow ───────────────────────────────────────────────────
    MWAA ==> INGEST
    INGEST ==> FEAT
    FEAT ==> KFP
    KFP ==> GATE
    GATE ==> PASS_NODE
    GATE ==> FAIL_NODE
    PASS_NODE ==> BATCH

    %% ── Secondary Connections ───────────────────────────────────────
    TRAIN -.-> MLFLOW
    REGISTER -.-> MLFLOW
    VALIDATE -.-> S3BUCK
    SPARK_FE -.-> S3BUCK
    INFER -.-> MLFLOW

    %% ── Styles ──────────────────────────────────────────────────────
    classDef aws fill:#232f3e,stroke:#ff9900,color:#ff9900,stroke-width:2px
    classDef task fill:#2b6cb0,stroke:#2c5282,color:#fff,stroke-width:1.5px
    classDef model fill:#276749,stroke:#1a4731,color:#fff,stroke-width:1.5px
    classDef gate fill:#fef3c7,stroke:#d97706,color:#92400e,stroke-width:2px
    classDef fail fill:#b91c1c,stroke:#991b1b,color:#fff,stroke-width:1.5px
    classDef serve fill:#6b21a8,stroke:#581c87,color:#fff,stroke-width:1.5px
    classDef storage fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:1.5px

    class SCHED aws
    class S3SENSOR,VALIDATE,SPARK_FE,FEAST task
    class TRAIN,EVAL,REGISTER model
    class GATE gate
    class FAIL_NODE fail
    class PASS_NODE model
    class INFER,REDIS,S3OUT,OSEARCH serve
    class S3BUCK,MLFLOW storage

    style MWAA fill:#f8fafc,stroke:#ff9900,stroke-width:2px,rx:10
    style INGEST fill:#f0f9ff,stroke:#2b6cb0,stroke-width:1.5px,rx:10
    style FEAT fill:#f0fff4,stroke:#276749,stroke-width:1.5px,rx:10
    style KFP fill:#f0fff4,stroke:#276749,stroke-width:1.5px,rx:10
    style BATCH fill:#faf5ff,stroke:#6b21a8,stroke-width:1.5px,rx:10
    style STORE fill:#f1f5f9,stroke:#1e3a5f,stroke-width:1.5px,rx:10
```

### 10.b Arquitectura de la Solución Completa (AWS)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD

    %% ── APP LAYER ──────────────────────────────────────────────
    APP["<b>DSRPMart Mobile / Web App</b><br/>React Native · React + Next.js"]
    GW["<b>Amazon API Gateway + AWS WAF</b><br/>Auth: Cognito · Rate Limit · HTTPS"]

    APP ==> GW

    %% ── SERVING LAYER ──────────────────────────────────────────
    subgraph SERVING["  SERVING LAYER  "]
        direction LR
        REC["<b>Recommendations Service</b><br/>FastAPI · EKS Pod<br/>GET /recs/:user_id<br/>Redis lookup → user:{id}:recs"]
        SEARCH["<b>Search Service</b><br/>FastAPI · EKS Pod<br/>GET /search?q=…<br/>BM25 + KNN retrieval · LightGBM ranking"]
    end

    GW ==> REC
    GW ==> SEARCH

    %% ── CACHE + INDEX ──────────────────────────────────────────
    subgraph CACHE_IDX["  CACHE + INDEX  "]
        direction LR
        REDIS["<b>Amazon ElastiCache — Redis 7.x</b><br/>3 shards · 2 replicas each<br/>Recs: user:{id}:recs → JSON TTL=6h<br/>Search: query:{hash} → JSON TTL=30min<br/>Feast: feast:online:{entity} → features"]
        OS["<b>Amazon OpenSearch Service</b><br/>Index: products-v{N}<br/>BM25 text + KNN 384d HNSW<br/>Blue-green swap · 3× r6g.xlarge"]
    end

    REC ==> REDIS
    SEARCH ==> REDIS
    SEARCH -.->|KNN queries| OS

    %% ── BATCH INFERENCE ────────────────────────────────────────
    subgraph BATCH["  BATCH INFERENCE  "]
        SPARK["<b>Spark Batch Inference</b><br/>Spark on EKS · Spark Operator<br/>Scheduled 4×/day · Spot instances<br/>Reads MLflow model + Feast features<br/>Writes → Redis + S3"]
    end

    SPARK ==>|writes| REDIS
    SPARK -.->|index update| OS

    %% ── EKS CLUSTER ────────────────────────────────────────────
    subgraph EKS["  Amazon EKS Cluster v1.29 — Managed Node Groups + Karpenter  "]
        direction TB
        subgraph ROW1["  "]
            direction LR
            NS_KF["<b>kubeflow</b><br/>KFP Pipelines · Katib HPO<br/>Notebooks"]
            NS_AF["<b>airflow</b><br/>MWAA Agent<br/>DAG sync from S3"]
            NS_ML["<b>mlflow</b><br/>Tracking Server · Model Registry<br/>Backend: RDS PostgreSQL<br/>Artifacts: S3"]
        end
        subgraph ROW2["  "]
            direction LR
            NS_FE["<b>feast</b><br/>Feast Server<br/>Online: Redis · Offline: S3"]
            NS_SP["<b>spark</b><br/>Spark Operator<br/>Driver/Exec · Spot pools"]
            NS_MO["<b>monitoring</b><br/>Prometheus · Grafana<br/>Alertmanager"]
        end
        subgraph ROW3["  "]
            direction LR
            NS_SV["<b>serving</b><br/>Rec Service · Search Svc"]
            NS_AR["<b>argocd</b><br/>GitOps Controller<br/>Syncs dsrpmart-infra repo<br/>Auto-heal · Rollback"]
        end
    end

    EKS -.-> SERVING
    EKS -.-> BATCH

    %% ── DATA LAYER ─────────────────────────────────────────────
    subgraph DATA["  DATA LAYER  "]
        direction LR
        S3["<b>Amazon S3</b><br/>raw/ · validated/ · processed/<br/>features/ · models/"]
        RS["<b>Amazon Redshift</b><br/>Serverless<br/>DWH + A/B analysis"]
        KIN["<b>Amazon Kinesis</b><br/>Data Streams<br/>user_events · search_logs<br/>Firehose → S3"]
        RDS["<b>Amazon RDS</b><br/>PostgreSQL<br/>Catálogo de productos"]
        DMS["<b>AWS DMS</b><br/>CDC → S3<br/>Catálogo sync"]
        GLUE["<b>AWS Glue</b><br/>Data Catalog<br/>Schema discovery"]
    end

    EKS ==> DATA
    SPARK -.-> S3

    %% ── SECURITY BAR ───────────────────────────────────────────
    SEC["<b>SECURITY LAYER</b><br/>IRSA · Secrets Manager · KMS · VPC · WAF · ECR (Trivy + cosign) · CloudTrail"]

    DATA ~~~ SEC

    %% ── STYLES ─────────────────────────────────────────────────
    classDef app fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px
    classDef aws fill:#232f3e,stroke:#ff9900,color:#ff9900,stroke-width:2px
    classDef svc fill:#6b21a8,stroke:#581c87,color:#fff,stroke-width:1.5px
    classDef redis fill:#dc2626,stroke:#b91c1c,color:#fff,stroke-width:1.5px
    classDef opensearch fill:#1e40af,stroke:#1e3a8a,color:#fff,stroke-width:1.5px
    classDef batch fill:#2b6cb0,stroke:#2c5282,color:#fff,stroke-width:1.5px
    classDef ns fill:#f8fafc,stroke:#94a3b8,color:#334155,stroke-width:1px
    classDef data fill:#276749,stroke:#1a4731,color:#fff,stroke-width:1.5px
    classDef security fill:#92400e,stroke:#78350f,color:#fef3c7,stroke-width:2px

    class APP app
    class GW aws
    class REC,SEARCH svc
    class REDIS redis
    class OS opensearch
    class SPARK batch
    class NS_KF,NS_AF,NS_ML,NS_FE,NS_SP,NS_MO,NS_SV,NS_AR ns
    class S3,RS,KIN,RDS,DMS,GLUE data
    class SEC security

    style SERVING fill:#faf5ff,stroke:#6b21a8,stroke-width:1.5px,rx:10
    style CACHE_IDX fill:#fef2f2,stroke:#dc2626,stroke-width:1.5px,rx:10
    style BATCH fill:#eff6ff,stroke:#2b6cb0,stroke-width:1.5px,rx:10
    style EKS fill:#f1f5f9,stroke:#ff9900,stroke-width:2px,rx:10
    style ROW1 fill:transparent,stroke:none
    style ROW2 fill:transparent,stroke:none
    style ROW3 fill:transparent,stroke:none
    style DATA fill:#f0fff4,stroke:#276749,stroke-width:1.5px,rx:10
```

### 10.c CI/CD Despliegue de Modelo

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD

    %% ── DEV ──────────────────────────────────────────────────
    subgraph DEV["🧑‍💻 DEV — Data Scientist / ML Engineer"]
        direction TB
        D1["<b>Branch</b><br/>feature/model-improvement-v2"]
        D2["<b>Experiment</b><br/>JupyterHub · Kubeflow Notebooks on EKS"]
        D3["<b>Track</b><br/>MLflow runs: experiment, params, metrics"]
        D4["<b>Version Data</b><br/>DVC add → datasets en S3"]
        D5["<b>Push</b><br/>git commit + push → PR a develop<br/>PR: run_id, métricas, changelog"]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    %% ── CI ───────────────────────────────────────────────────
    subgraph CI["⚙️ CI — GitHub Actions on PR"]
        direction TB
        CI1["<b>Code Quality</b><br/>ruff check · mypy · bandit"]
        CI2["<b>Tests</b><br/>pytest 80 % cov · integration moto<br/>Great Expectations data contracts"]
        CI3["<b>Reproducibility</b><br/>dvc repro --dry · dvc diff"]
        CI4(["✅ All checks pass → Ready for review"])
        CI1 --> CI2 --> CI3 --> CI4
    end

    %% ── CD STAGING ───────────────────────────────────────────
    subgraph CD_STG["🟡 CD STAGING — on merge develop"]
        direction TB
        S1["<b>Build & Push</b><br/>Docker → ECR · Trivy scan · cosign sign"]
        S2["<b>Deploy Staging</b><br/>Helm values-staging · ArgoCD auto-sync"]
        S3["<b>ML Pipeline Staging</b><br/>KFP train · Metrics vs Champion<br/>NDCG, MRR, HitRate, Coverage"]
        S4["<b>Validate</b><br/>Evidently drift report → S3<br/>MLflow stage = Staging"]
        S5(["✅ Metrics PASS → promote"])
        S6(["❌ Metrics FAIL → Slack alert, block"])
        S1 --> S2 --> S3 --> S4
        S4 --> S5
        S4 -.-> S6
    end

    %% ── CD PRODUCTION ────────────────────────────────────────
    subgraph CD_PRD["🟢 CD PRODUCTION — on merge main"]
        direction TB
        P1["<b>Tag & Push</b><br/>:sha · :semver · GitHub Release"]
        subgraph ARGO["ArgoCD GitOps Controller"]
            direction TB
            A1["Detect change in dsrpmart-infra/prod"]
            A2["Rolling update pods — zero downtime"]
            A3["readiness + liveness probes"]
            A4["Canary rollout 10 % → 50 % → 100 %"]
            A5["Slack #ml-deployments ✅"]
            A1 --> A2 --> A3 --> A4 --> A5
        end
        P2["<b>Post-Deploy</b><br/>Smoke tests /recs & /search<br/>MLflow Staging → Production<br/>Shadow mode Challenger · Dashboards"]
        P1 --> ARGO --> P2
    end

    %% ── ROLLBACK ─────────────────────────────────────────────
    subgraph RB["🔴 ROLLBACK AUTOMÁTICO"]
        direction TB
        R1["<b>Triggers</b><br/>error_rate > 1 % (5 min)<br/>latency p99 > SLA (10 min)<br/>Manual: ArgoCD UI"]
        R2["<b>Actions</b><br/>git revert values-prod → prev SHA<br/>K8s pods rollback · MLflow revert<br/>Redis rewrite predictions<br/>PagerDuty on-call + runbook"]
        R3["Post-incident: blameless post-mortem 48 h"]
        R1 --> R2 --> R3
    end

    %% ── MAIN FLOW ────────────────────────────────────────────
    D5 ==> CI
    CI4 ==>|"PR approved + merge develop"| S1
    S5 ==>|"Release PR → merge main"| P1
    P2 -.->|"health check fail / alert"| R1

    %% ── STYLES ───────────────────────────────────────────────
    classDef dev  fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px
    classDef ci   fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:1px
    classDef cd_stg fill:#ca8a04,stroke:#a16207,color:#fff,stroke-width:1px
    classDef cd_prd fill:#15803d,stroke:#166534,color:#fff,stroke-width:2px
    classDef rb   fill:#b91c1c,stroke:#991b1b,color:#fff,stroke-width:1px

    class D1,D2,D3,D4,D5 dev
    class CI1,CI2,CI3,CI4 ci
    class S1,S2,S3,S4,S5,S6 cd_stg
    class P1,P2,A1,A2,A3,A4,A5 cd_prd
    class R1,R2,R3 rb

    style DEV    fill:#0f172a,stroke:#1e3a5f,stroke-width:2px,rx:10,color:#e2e8f0
    style CI     fill:#eff6ff,stroke:#2563eb,stroke-width:2px,rx:10,color:#1e3a8a
    style CD_STG fill:#fefce8,stroke:#ca8a04,stroke-width:2px,rx:10,color:#713f12
    style CD_PRD fill:#f0fdf4,stroke:#15803d,stroke-width:2px,rx:10,color:#14532d
    style ARGO   fill:#dcfce7,stroke:#166534,stroke-width:1px,rx:8,color:#14532d
    style RB     fill:#fef2f2,stroke:#b91c1c,stroke-width:2px,rx:10,color:#7f1d1d
```

---

## 11. Monitoreo, Data Drift y Observabilidad

> **¿Por qué monitorear?** En una aplicación de ventas, si un modelo se degrada (por ejemplo, porque cambian las tendencias de compra o llegan nuevos productos), la conversión baja sin que nadie lo note a tiempo. El monitoreo continuo detecta estos cambios automáticamente y puede disparar un reentrenamiento del modelo para que siempre esté actualizado.

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

#### Detección — Evidently AI · Batch Reports

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '12px', 'fontFamily': 'Segoe UI, Arial', 'lineColor': '#64748b'}}}%%
flowchart TD
    DAG["<b>Airflow DAG</b><br>data_drift_monitoring<br>daily 04:00 UTC"]:::dag

    REF[("Reference Data<br>Training dataset<br>feature distributions")]:::source
    CUR[("Current Data<br>Last 24 h production<br>features · Feast offline")]:::source

    REF --> T1
    CUR --> T1

    T1["<b>Task 1 — Compute Feature Drift</b><br>PSI · KS Test · Wasserstein Distance"]:::task
    T1 --> RPT["Evidently HTML Report<br>→ S3 + MLflow run"]:::artifact

    T1 --> T2["<b>Task 2 — Compute Prediction Drift</b><br>Score distribution T vs T-7<br>Top-K product_id PSI<br>Alerta si &gt;20 % items nuevos en Top-10"]:::task

    T2 --> T3["<b>Task 3 — Compute Target Drift</b><br>CTR real vs esperado (cohort semanal)<br>Conversion rate trend (rolling 7 d)<br>Alerta si gap &gt;2 σ vs media histórica"]:::task

    T3 --> T4{"<b>Task 4 — Decision</b><br>Evaluar max PSI"}:::gate

    T4 -->|"PSI ≥ 0.25<br>Drift crítico"| RETRAIN["Trigger Retrain DAG<br>Kubeflow Pipeline"]:::retrain
    T4 -->|"0.10 ≤ PSI &lt; 0.25<br>Drift moderado"| WARN["Slack Warning<br>#ml-alerts"]:::warning
    T4 -->|"PSI &lt; 0.10<br>Sin drift"| OK["Log OK<br>Continuar schedule normal"]:::ok

    DAG -.-> T1

    classDef dag fill:#1e3a5f,stroke:#0f2942,color:#e2e8f0,stroke-width:2px,font-weight:bold
    classDef source fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1px
    classDef task fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:2px
    classDef artifact fill:#f8fafc,stroke:#94a3b8,color:#334155,stroke-width:1px,stroke-dasharray:5 5
    classDef gate fill:#fef3c7,stroke:#d97706,color:#92400e,stroke-width:3px,font-weight:bold
    classDef retrain fill:#dc2626,stroke:#b91c1c,color:#fff,stroke-width:2px,font-weight:bold
    classDef warning fill:#f59e0b,stroke:#d97706,color:#fff,stroke-width:2px
    classDef ok fill:#16a34a,stroke:#15803d,color:#fff,stroke-width:2px
```

#### Métricas Monitoreadas por Modelo

**Productos Recomendados:**

| Feature | Test |
|---|---|
| `user_ctr_by_category` | PSI |
| `session_length` | KS test |
| `avg_price_viewed` | Wasserstein |
| `category_distribution` | Chi-squared |
| `prediction_score_distribution` | PSI |

**Motor de Búsqueda:**

| Feature | Test |
|---|---|
| `query_length_distribution` | KS test |
| `query_category_distribution` | Chi-squared |
| `embedding_centroid_shift` | Cosine distance |
| `retrieval_score_distribution` | PSI |
| `zero_result_rate_trend` | Statistical process control |

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
| 1 | Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). *Machine Learning Operations (MLOps): Overview, Definition, and Architecture.* IEEE Access. | Paper fundacional que define los niveles de madurez MLOps 0-2 utilizados en este documento | <https://arxiv.org/abs/2205.02302> |
| 2 | Covington, P., Adams, J., & Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations.* RecSys. | Arquitectura Two-Tower original de Google que inspira nuestro Stage A de retrieval | <https://research.google/pubs/pub45530/> |
| 3 | Yi, X. et al. (2019). *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations.* RecSys. | In-batch sampled softmax y corrección de bias que usamos en el Two-Tower | <https://research.google/pubs/pub48840/> |
| 4 | Burges, C.J.C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research. | Fundamento teórico de LambdaRank implementado vía LightGBM en ambos modelos | <https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/> |
| 5 | Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. | Base del modelo Sentence-BERT utilizado para embeddings de búsqueda | <https://arxiv.org/abs/1908.10084> |
| 6 | Barkan, O. & Koenigstein, N. (2016). *Item2Vec: Neural Item Embedding for Collaborative Filtering.* IEEE MLSP. | Método de embeddings Item2Vec aplicado a secuencias de sesión | <https://arxiv.org/abs/1603.04259> |
| 7 | Carbonell, J. & Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR. | Maximal Marginal Relevance para diversificación de resultados en re-ranking | <https://dl.acm.org/doi/10.1145/290941.291025> |
| 8 | Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. | Anti-patrones de ML en producción que esta arquitectura evita | <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html> |

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
| 1 | MLflow Documentation | <https://mlflow.org/docs/latest/index.html> |
| 2 | Kubeflow Pipelines on AWS EKS | <https://www.kubeflow.org/docs/distributions/aws/> |
| 3 | Apache Airflow – Amazon MWAA | <https://docs.aws.amazon.com/mwaa/> |
| 4 | Feast Feature Store | <https://docs.feast.dev/> |
| 5 | Evidently AI (Data Drift) | <https://docs.evidentlyai.com/> |
| 6 | Amazon EKS Best Practices | <https://aws.github.io/aws-eks-best-practices/> |
| 7 | ArgoCD GitOps for K8s | <https://argo-cd.readthedocs.io/> |
| 8 | Amazon OpenSearch KNN Plugin | <https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html> |
| 9 | Sentence-Transformers Library | <https://www.sbert.net/> |
| 10 | LightGBM (Learning-to-Rank) | <https://lightgbm.readthedocs.io/en/stable/> |
| 11 | Spark on Kubernetes (Spark Operator) | <https://github.com/kubeflow/spark-operator> |
| 12 | Terraform AWS Provider | <https://registry.terraform.io/providers/hashicorp/aws/latest/docs> |
| 13 | Karpenter (K8s Autoscaler) | <https://karpenter.sh/docs/> |
| 14 | Great Expectations (Data Quality) | <https://docs.greatexpectations.io/> |
| 15 | FAISS (Facebook AI Similarity Search) | <https://github.com/facebookresearch/faiss> |
| 16 | DVC – Data Version Control | <https://dvc.org/doc> |
| 17 | Amazon Kinesis Data Streams | <https://docs.aws.amazon.com/streams/latest/dev/> |
| 18 | Amazon Redshift Serverless | <https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-whatis.html> |
| 19 | Prometheus + Grafana Stack | <https://prometheus.io/docs/> / <https://grafana.com/docs/> |
| 20 | Trivy (Container Security) | <https://trivy.dev/latest/docs/> |

### 12.4 Blogs y Recursos Técnicos de Industria

| # | Referencia | URL |
|---|---|---|
| 1 | Model Cards for Model Reporting (Google) | <https://modelcards.withgoogle.com/> |
| 2 | Kaggle – Model Cards Template | <https://www.kaggle.com/code/var0101/model-cards> |
| 3 | ML Design Docs – Eugene Yan | <https://eugeneyan.com/writing/ml-design-docs/> |
| 4 | Software Engineering RFC and Design – Pragmatic Engineer | <https://newsletter.pragmaticengineer.com/p/software-engineering-rfc-and-design> |
| 5 | Champion/Challenger Pattern for ML Deployment | <https://christophergs.com/machine%20learning/2019/03/30/deploying-and-versioning-data-science-models-at-scale/> |
| 6 | Spotify – ML Infrastructure Case Study | <https://engineering.atspotify.com/2019/12/13/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow/> |
| 7 | Netflix – System Architectures for Personalization and Recommendation | <https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8> |
| 8 | Uber – Michelangelo ML Platform | <https://www.uber.com/blog/michelangelo-machine-learning-platform/> |
| 9 | Airbnb – Machine Learning Infrastructure | <https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d> |
| 10 | AWS Well-Architected ML Lens | <https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html> |

---

*Documento elaborado como Proyecto Final del Curso IV – Especialización Machine Learning Engineering · Febrero 2026*  
*Herramientas recomendadas para pasar los diagramas de texto a formato visual: [Excalidraw](https://excalidraw.com/) · [draw.io](https://draw.io/) · [tldraw](https://www.tldraw.com/) · [Miro](https://miro.com/) · [Lucidchart](https://lucidchart.com/)*
