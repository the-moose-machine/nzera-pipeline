# nzera-pipeline
This pipeline is used for implementing and maintaining a data warehouse that is used for the maintenance of a CNN-based model. It has been produced to be a part of an automated chatbot that provides legal advice to users an matters relatiing to past, current and potential cases cases with the New Zealand Employment Relations Authority.

This pipeline has the following stages:
1. Testing to assess whether new documents have by made available by the NZERA. 
2. Extracting the cases.
3. Transforming the cases by extracting features and individual paragraphs
4. Obtaining the outcome of the new case by passing it through a pre-trained model
5. Allocating the label as determined by the model.

It is recommended that the accuracy of the predictions be manually assessed by a Subject Matter Expert periodically.

This commit is a prototype, needs some bug-fixing and cannot yet be considered suitable for implementation.
