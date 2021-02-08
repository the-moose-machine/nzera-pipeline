# nzera-pipeline
This pipeline is used for implementing and maintaining a data warehouse that is used for the maintenance of a RNN-based model. It has been produced to be a part of an automated chatbot that provides legal advice to users an matters relatiing to past, current and potential cases cases with the New Zealand Employment Relations Authority.

## Pipeline Stages
This pipeline has the following stages:
1. Testing to assess whether new documents have by made available by the NZERA. 
2. Extracting the cases.
2. Load the cases into a Mongo database.
3. Transforming the cases by extracting features and individual paragraphs
4. Obtaining the outcome of the new case by passing it through a pre-trained model
5. Allocating the label as determined by the model.
6. Saving the new document and label into the Mongo database.

It is recommended that the accuracy of the predictions be manually assessed by a Subject Matter Expert periodically.

## Next stage of the pipeline
1. Test with new unlabelled document.
1. Re-train the RNN model periodically based on the newly acquired data.

## Known issues
1. This commit is a prototype, needs some bug-fixing and cannot yet be considered suitable for implementation.
2. It has not been written in PEP 8 format and is not yet suitable for deployment.
3. Includes journal entries and the remarks are not user-friendly
4. Variables have not been initialised in a user-friendly manner.
