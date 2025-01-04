# API Reference

- ::: readme_ready.index.index
    handler: python
    options:
      show_root_heading: true
      show_source: false
      separate_signature: true

- ::: readme_ready.index.convert_json_to_markdown
    handler: python
    options:
      show_root_heading: true
      show_source: false
      separate_signature: true

- ::: readme_ready.index.create_vector_store
    handler: python
    options:
      show_root_heading: true
      show_source: false
      separate_signature: true

- ::: readme_ready.index.process_repository
    handler: python
    options:
      members:
        - process_repository
      show_root_heading: true
      show_source: false
      separate_signature: true
      
- ::: readme_ready.query.query
    handler: python
    options:
      members:
        - query
        - generate_readme
      show_root_heading: true
      show_source: false
      separate_signature: true

- ::: readme_ready.query.create_chat_chain
    handler: python
    options:
      members:
        - make_qa_chain
        - make_readme_chain
      show_root_heading: true
      show_source: false
      separate_signature: true

- ::: readme_ready.types
    handler: python
    options:
      members:
        - AutodocReadmeConfig
        - AutodocRepoConfig
        - AutodocUserConfig
        - LLMModels
      show_root_heading: true
      show_source: false
      separate_signature: true