from readme_ready.utils.file_utils import (
    get_file_name,
    github_file_url,
    github_folder_url,
)


def test_get_file_name_with_delimiter():
    assert get_file_name("example.txt") == "example.md"


def test_get_file_name_without_delimiter():
    assert get_file_name("example") == "example.md"


def test_get_file_name_custom_delimiter():
    assert get_file_name("example-text", delimiter="-") == "example.md"


def test_get_file_name_no_delimiter_custom_extension():
    assert get_file_name("example", extension=".txt") == "example.txt"


def test_get_file_name_with_delimiter_custom_extension():
    assert get_file_name("example.txt", extension=".txt") == "example.txt"


def test_get_file_name_with_multiple_delimiters():
    assert get_file_name("my.example.txt") == "my.example.md"


def test_get_file_name_with_no_delimiter_and_no_extension():
    assert get_file_name("example", extension="") == "example"


def test_get_file_name_with_delimiter_and_no_extension():
    assert get_file_name("example.txt", extension="") == "example"


def test_get_file_name_empty_input():
    assert get_file_name("") == ".md"


def test_get_file_name_delimiter_not_in_input():
    assert get_file_name("example", delimiter="/") == "example.md"


def test_get_file_name_delimiter_at_end():
    assert get_file_name("example.", delimiter=".") == "example.md"


def test_get_file_name_delimiter_at_start():
    assert get_file_name(".example", delimiter=".") == ".md"


def test_get_file_name_delimiter_multiple_occurrences():
    assert get_file_name("my.file.name.txt") == "my.file.name.md"


def test_github_file_url_link_hosted_true():
    github_root = "https://github.com/user/repo"
    input_root = "/home/user/project"
    file_path = "/home/user/project/docs/file.md"
    link_hosted = True
    expected_url = f"{github_root}/{file_path[len(input_root)-1:]}"
    assert (
        github_file_url(github_root, input_root, file_path, link_hosted)
        == expected_url
    )


def test_github_file_url_link_hosted_false():
    github_root = "https://github.com/user/repo"
    input_root = "/home/user/project"
    file_path = "/home/user/project/docs/file.md"
    link_hosted = False
    expected_url = f"{github_root}/blob/master/{file_path[len(input_root)-1:]}"
    assert (
        github_file_url(github_root, input_root, file_path, link_hosted)
        == expected_url
    )


def test_github_file_url_empty_input_root():
    github_root = "https://github.com/user/repo"
    input_root = ""
    file_path = "/docs/file.md"
    link_hosted = False
    expected_url = f"{github_root}/blob/master/{file_path[-1:]}"
    assert (
        github_file_url(github_root, input_root, file_path, link_hosted)
        == expected_url
    )


def test_github_file_url_empty_file_path():
    github_root = "https://github.com/user/repo"
    input_root = "/home/user/project"
    file_path = ""
    link_hosted = False
    expected_url = f"{github_root}/blob/master/{file_path[len(input_root)-1:]}"
    assert (
        github_file_url(github_root, input_root, file_path, link_hosted)
        == expected_url
    )


def test_github_folder_url_link_hosted_true():
    github_root = "https://github.com/user/repo"
    input_root = "/home/user/project"
    folder_path = "/home/user/project/docs/"
    link_hosted = True
    expected_url = f"{github_root}/{folder_path[len(input_root)-1:]}"
    assert (
        github_folder_url(github_root, input_root, folder_path, link_hosted)
        == expected_url
    )


def test_github_folder_url_link_hosted_false():
    github_root = "https://github.com/user/repo"
    input_root = "/home/user/project"
    folder_path = "/home/user/project/docs/"
    link_hosted = False
    expected_url = (
        f"{github_root}/tree/master/{folder_path[len(input_root)-1:]}"
    )
    assert (
        github_folder_url(github_root, input_root, folder_path, link_hosted)
        == expected_url
    )


def test_github_folder_url_empty_input_root():
    github_root = "https://github.com/user/repo"
    input_root = ""
    folder_path = "/docs/"
    link_hosted = False
    expected_url = f"{github_root}/tree/master/{folder_path[-1:]}"
    assert (
        github_folder_url(github_root, input_root, folder_path, link_hosted)
        == expected_url
    )


def test_github_folder_url_empty_folder_path():
    github_root = "https://github.com/user/repo"
    input_root = "/home/user/project"
    folder_path = ""
    link_hosted = False
    expected_url = (
        f"{github_root}/tree/master/{folder_path[len(input_root)-1:]}"
    )
    assert (
        github_folder_url(github_root, input_root, folder_path, link_hosted)
        == expected_url
    )
