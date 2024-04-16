from pathlib import Path

# Import the necessary classes and functions from the files you provided
from your_module.convertJsonToMarkdown import convertJsonToMarkdown as convert
from your_module.preprocessRepository import processRepository as preprocess
from your_module.createVectorStore import createVectorStore as createStore

# Placeholder for the AutodocRepoConfig class, normally imported
class AutodocRepoConfig:
    def __init__(self, name, repositoryUrl, root, output, llms, priority,
                 maxConcurrentCalls, addQuestions, ignore, filePrompt,
                 folderPrompt, chatPrompt, contentType, targetAudience, linkHosted):
        self.name = name
        self.repositoryUrl = repositoryUrl
        self.root = root
        self.output = output
        self.llms = llms
        self.priority = priority
        self.maxConcurrentCalls = maxConcurrentCalls
        self.addQuestions = addQuestions
        self.ignore = ignore
        self.filePrompt = filePrompt
        self.folderPrompt = folderPrompt
        self.chatPrompt = chatPrompt
        self.contentType = contentType
        self.targetAudience = targetAudience
        self.linkHosted = linkHosted

# Function implementations
def processRepository(config):
    # This function will handle preprocessing of the repository
    preprocess(config)

def convertJsonToMarkdown(config):
    # This function will convert the JSON outputs into Markdown
    convert(config)

def createVectorStore(config):
    # This function will create a vector store from the processed data
    createStore(config)

# # Utility functions for UI feedback
# def updateSpinnerText(text):
#     print(text)  # Simplistic spinner text update simulation

# def spinnerSuccess():
#     print("Done!")  # Simplistic success message

def index(config: AutodocRepoConfig):
    json_path = Path(config.output) / 'docs' / 'json'
    markdown_path = Path(config.output) / 'docs' / 'markdown'
    data_path = Path(config.output) / 'docs' / 'data'

    # Ensure directories exist
    json_path.mkdir(parents=True, exist_ok=True)
    markdown_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Process the repository to create JSON files
    updateSpinnerText('Processing repository...')
    processRepository(AutodocRepoConfig(
        name=config.name,
        repositoryUrl=config.repositoryUrl,
        root=config.root,
        output=str(json_path),
        llms=config.llms,
        priority=config.priority,
        maxConcurrentCalls=config.maxConcurrentCalls,
        addQuestions=config.addQuestions,
        ignore=config.ignore,
        filePrompt=config.filePrompt,
        folderPrompt=config.folderPrompt,
        chatPrompt=config.chatPrompt,
        contentType=config.contentType,
        targetAudience=config.targetAudience,
        linkHosted=config.linkHosted,
    ))
    #spinnerSuccess()

    # Convert the JSON files to Markdown
    updateSpinnerText('Creating markdown files...')
    convertJsonToMarkdown(AutodocRepoConfig(
        name=config.name,
        repositoryUrl=config.repositoryUrl,
        root=str(json_path),
        output=str(markdown_path),
        llms=config.llms,
        priority=config.priority,
        maxConcurrentCalls=config.maxConcurrentCalls,
        addQuestions=config.addQuestions,
        ignore=config.ignore,
        filePrompt=config.filePrompt,
        folderPrompt=config.folderPrompt,
        chatPrompt=config.chatPrompt,
        contentType=config.contentType,
        targetAudience=config.targetAudience,
        linkHosted=config.linkHosted,
    ))
    #spinnerSuccess()

    # Create a vector store from the Markdown documents
    updateSpinnerText('Creating vector files...')
    createVectorStore(AutodocRepoConfig(
        name=config.name,
        repositoryUrl=config.repositoryUrl,
        root=str(markdown_path),
        output=str(data_path),
        llms=config.llms,
        priority=config.priority,
        maxConcurrentCalls=config.maxConcurrentCalls,
        addQuestions=config.addQuestions,
        ignore=config.ignore,
        filePrompt=config.filePrompt,
        folderPrompt=config.folderPrompt,
        chatPrompt=config.chatPrompt,
        contentType=config.contentType,
        targetAudience=config.targetAudience,
        linkHosted=config.linkHosted,
    ))
    #spinnerSuccess()

# Example usage
if __name__ == "__main__":
    asyncio.run(index(AutodocRepoConfig(
        name="ExampleProject",
        repositoryUrl="https://github.com/example/project",
        root="/path/to/project",
        output="/path/to/output",
        llms="OpenAIChatModel",
        priority=1,
        maxConcurrentCalls=5,
        addQuestions=True,
        ignore=["*.tmp", "*.log"],
        filePrompt="Describe this file:",
        folderPrompt="Describe this folder:",
        chatPrompt="Discuss this code:",
        contentType="code",
        targetAudience="developers",
        linkHosted=True
    )))
