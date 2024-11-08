def get_file_name(input_str, delimiter=".", extension=".md"):
    """Get File Name"""
    last_delimiter_index = input_str.rfind(delimiter)
    if last_delimiter_index == -1:
        # delimiter not found in string
        return input_str + extension
    else:
        return input_str[:last_delimiter_index] + extension


def github_file_url(github_root, input_root, file_path, link_hosted):
    """Get GitHub File URL"""
    if link_hosted:
        return f"{github_root}/{file_path[len(input_root) - 1:]}"
    else:
        return f"{github_root}/blob/master/{file_path[len(input_root) - 1:]}"


def github_folder_url(github_root, input_root, folder_path, link_hosted):
    """Get GitHub Folder URL"""
    if link_hosted:
        return f"{github_root}/{folder_path[len(input_root) - 1:]}"
    else:
        return f"{github_root}/tree/master/{folder_path[len(input_root) - 1:]}"
