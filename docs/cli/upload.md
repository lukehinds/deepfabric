# upload

The `upload` command publishes datasets to Hugging Face Hub with automatically generated metadata and dataset cards, streamlining the process of sharing synthetic datasets with the broader machine learning community. This integration handles authentication, file upload, and documentation generation through a single command.

The upload process creates comprehensive dataset documentation that includes generation methodology, model information, and usage guidelines, ensuring that published datasets provide sufficient context for effective reuse.

## Basic Usage

Upload a dataset file to Hugging Face Hub:

```bash
deepfabric upload dataset.jsonl --repo username/dataset-name
```

This command uploads the dataset file and creates a dataset card with automatically generated metadata about the content and generation process.

## Authentication Methods

The upload command supports multiple authentication approaches for flexibility across different environments:

**Environment Variable** provides the most secure approach for production environments:

```bash
export HF_TOKEN="your-huggingface-token"
deepfabric upload dataset.jsonl --repo username/dataset-name
```

**Command Line Option** enables token specification directly in the command:

```bash
deepfabric upload dataset.jsonl --repo username/dataset-name --token your-token
```

**Hugging Face CLI** authentication works automatically if you've previously authenticated with the Hugging Face CLI:

```bash
huggingface-cli login
deepfabric upload dataset.jsonl --repo username/dataset-name
```

## Repository Management

The upload command handles repository creation and updates automatically:

**New Repositories** are created automatically when uploading to non-existent repositories, using the dataset filename as the initial commit message.

**Existing Repositories** receive updates to both the dataset files and the dataset card, with commit messages indicating the update source.

**Repository Naming** follows Hugging Face conventions with the format `username/dataset-name` or `organization/dataset-name`.

## Dataset Tagging

Customize dataset discoverability through tag specification:

```bash
deepfabric upload dataset.jsonl \
  --repo username/educational-content \
  --tags educational \
  --tags programming \
  --tags synthetic
```

**Automatic Tags** are added by default including "deepfabric" and "synthetic" to identify the generation method and nature of the data.

**Custom Tags** enhance discoverability by specifying domain, use case, or content characteristics relevant to your specific dataset.

**Tag Strategy** should balance specificity with searchability to maximize appropriate dataset discovery.

## Generated Documentation

The upload process creates comprehensive dataset cards that include:

**Generation Metadata** documenting the DeepFabric version, configuration parameters, and model providers used in dataset creation.

**Content Statistics** providing information about dataset size, topic coverage, and example formats.

**Usage Guidelines** offering suggestions for appropriate use cases and potential limitations of the synthetic data.

**Licensing Information** applying appropriate licenses based on the generation methodology and intended use patterns.

## File Organization

The upload process organizes files according to Hugging Face Hub conventions:

```
repository-name/
├── README.md          # Generated dataset card
├── dataset.jsonl      # Your uploaded dataset
└── .gitattributes     # LFS configuration for large files
```

Large dataset files are automatically configured for Git LFS to ensure efficient storage and retrieval.

## Integration with Generation

The upload command integrates seamlessly with the generation workflow:

```bash
# Generate dataset
deepfabric generate config.yaml

# Upload immediately after generation
deepfabric upload training_dataset.jsonl --repo myorg/new-dataset
```

This pattern enables immediate sharing of generated datasets while maintaining clear provenance information.

## Batch Upload Operations

Upload multiple related datasets to maintain organized dataset collections:

```bash
# Upload training and validation sets
deepfabric upload train_dataset.jsonl --repo myorg/comprehensive-dataset --tags training
deepfabric upload val_dataset.jsonl --repo myorg/comprehensive-dataset --tags validation
```

This approach creates dataset repositories with multiple related files and appropriate metadata for each component.

## Error Handling

The upload process includes comprehensive error handling for common issues:

**Authentication Failures** provide clear guidance for token setup and repository permissions.

**Network Issues** implement retry logic for transient connectivity problems.

**Repository Conflicts** offer guidance for resolving naming conflicts and access issues.

**File Format Problems** validate dataset format before upload to prevent corrupted uploads.

## Privacy and Security

Consider privacy implications when uploading synthetic datasets:

**Content Review** is recommended before publishing to ensure generated content meets your organization's publication standards.

**Access Control** can be managed through Hugging Face Hub repository privacy settings if needed.

**Attribution Requirements** are handled automatically through the generated dataset cards and metadata.

## Repository Updates

Subsequent uploads to existing repositories update both data and metadata:

**Incremental Updates** preserve existing repository history while adding new content and updating documentation.

**Version Management** through Hugging Face Hub's built-in versioning supports dataset evolution and improvement tracking.

**Change Documentation** automatically updates dataset cards to reflect changes in generation methodology or content characteristics.

??? tip "Publication Best Practices"
    Review generated dataset cards before publication and consider adding domain-specific usage notes or limitations. Use descriptive repository names and comprehensive tagging to maximize appropriate discovery and usage of your synthetic datasets.
