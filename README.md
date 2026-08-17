\# GNN Streaming Fraud Detection



A Graph Neural Network (GNN) based fraud detection proof of concept for financial transaction networks using the PaySim dataset.



\## Project Overview



Traditional fraud detection models generally analyze transactions as independent rows.



This project represents the financial transaction system as a graph:



\- Accounts are represented as nodes.

\- Transactions are represented as directed edges.

\- Transaction attributes are represented as edge features.

\- Historical account behavior is represented as node features.



A temporal GraphSAGE model is used to rank future transactions according to their fraud risk.



\## Architecture



```text

PaySim Transactions

&#x20;       |

&#x20;       v

Data Preprocessing

&#x20;       |

&#x20;       v

Transaction Graph

&#x20;       |

&#x20;       +-------------------+

&#x20;       |                   |

&#x20;       v                   v

&#x20;  Node Features      Edge Features

&#x20;       |                   |

&#x20;       +---------+---------+

&#x20;                 |

&#x20;                 v

&#x20;          Temporal GNN

&#x20;                 |

&#x20;                 v

&#x20;         Fraud Risk Score

&#x20;                 |

&#x20;                 v

&#x20;      Transaction Ranking

&#x20;                 |

&#x20;                 v

&#x20;       Analyst Investigation

