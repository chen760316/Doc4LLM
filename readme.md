**`datasets/`** Contains the source data for all datasets used in this paper.

**`doc_to_triplets/`** Introduces a simple and practical method for extracting **triplets** (excluding attributes) from documents. If your work does not require the granularity of **Attribute Graphs**, you can refer to the artifacts here for converting documents into **RDF graphs**.

**`DocChecker_and_exps/`** Detailed implementation and evaluation of **DocChecker**:

- **`DocChecker/`**: Contains the specific implementation of DocChecker (including the **Detector**, **Learner**, and other components) across the 15 datasets mentioned in the paper.
- **`experiments/`**: Includes the experimental code discussed in the paper.
- **`rule_discovery_inter_doc/`**: Focuses on the application of DocChecker for **cross-document contradiction detection**.
- **`rule_discovery_intra_doc/`**: Focuses on the application of DocChecker for **intra-document contradiction detection**.
- **`downstream_tasks/`**: Explores potential future applications of DocChecker within **ML, NLP, and RAG (Retrieval-Augmented Generation)** domains.
- **`rule_predicates/`**: Contains the **predicate definitions** for RICO rules.