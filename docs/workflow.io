%% Internal DeepFund workflow diagram.
%% This shows the `decision_making` path, not the API wrapper.

flowchart TD
    A[Data t-1] --> |News Summary| B[News Analyst]
    B[News Analyst] --> |Tool| L[Sentiment Classification]
    L[Sentiment Classification]  --> |Signal: B/H/S| G[Analyst Signals]

    A[Data t-1] --> |Price History| C[Technical Analyst]

    C[Technical Analyst] --> |Tool| M[Technical Indicators]
    M[Technical Indicators] --> |Signal: B/H/S| G[Analyst Signals]


    G[Analyst Signals] --> |Signals: B/H/S| D[Risk Manager]
    H[Portfolio] --> |State| D[Risk Manager]
    D[Risk Manager] --> |New Positioning| I[Position Verifier]
    I[Position Verifier] --> |Verified New Positioning| J[Shares Calculation]

    H[Portfolio] --> |Current Shares| E[Portfolio Manager]
    K[Database] --> |Decision Memories t-n| E[Portfolio Manager]
    A[Data t-1] --> |Current Price| E[Portfolio Manager]
    J[Shares Calculation] --> |Tradeable Shares| E[Portfolio Manager]

    E[Portfolio Manager] --> |Decision t| K[Database]

classDef input fill:#1f2937,color:#fff
classDef signal fill:#374151,color:#fff
classDef risk fill:#4b5563,color:#fff
classDef execution fill:#6b7280,color:#fff
classDef memory fill:#111827,color:#fff


graph TD

%% ============ INPUT LAYER ============
subgraph Input Layer
    A[Data t-1]
end

%% ============ SIGNAL LAYER ============
subgraph Signal Analyst Layer
    A -->|News| B[News Analyst]
    A -->|Price History| C[Technical Analyst]
    B --> D[Sentiment Model]
    C --> E[Technical Indicators]
    D --> F[Analyst Signals]
    E --> F
end

%% ============ Portfolio State ============
subgraph Portfolio
    P[Portfolio State]
end
%% ============ RISK LAYER ============
subgraph Risk Analyst Layer
    F --> G[Risk Manager]
    P[Portfolio State] --> G
    G --> H[Position Verifier]
    H --> I[Shares Calculation]
end

%% ============ EXECUTION LAYER ============
subgraph Portfolio Manager Layer
    I --> J[Portfolio Manager]
    P --> J
    A[Current Price] --> J
end

%% ============ MEMORY ============
subgraph Memory
    J --> K[(Database)]
end

%% ======= STYLE DEFINITIONS =======
classDef input fill:#1f2937,color:#fff,stroke:#000
classDef signal fill:#374151,color:#fff,stroke:#000
classDef risk fill:#4b5563,color:#fff,stroke:#000
classDef execution fill:#6b7280,color:#fff,stroke:#000
classDef memory fill:#111827,color:#fff,stroke:#000

%% ======= APPLY CLASSES =======
class A,P,CP input
class B,C,D,E,F signal
class G,H,I risk
class J execution
class K memory
