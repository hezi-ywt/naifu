from hfutils.operate import download_file_to_file, download_archive_as_directory, download_directory_as_directory

# # Download a single file from the repository
# download_file_to_file(
#     local_file='/nieta/soso/乙女美男.zip',
#     repo_id='heziiiii/soso',
#     file_in_repo='乙女美男.zip'
# )

# # Download an archive file from the repository and extract it to the given directory
# # More formats of archive files are supported
# # See: https://deepghs.github.io/hfutils/main/api_doc/archive/index.html
# download_archive_as_directory(
#     local_directory='/your/local/directory',
#     repo_id='your/repository',
#     file_in_repo='archive/file/in/your/repo.zip',
# )

# Download files from the repository as a directory tree
download_directory_as_directory(
    local_directory='/nieta/soso/wlop大神鬼刀_4k_filtered_webp',
    repo_id='heziiiii/soso',
    repo_type='dataset',
    dir_in_repo='wlop大神鬼刀_4k_filtered_webp'
)
# download_directory_as_directory(
#     local_directory='/root/autodl-tmp/soso/danbooru_images_hy_artist_json_packed',
#     repo_id='heziiiii/soso',
#     dir_in_repo='danbooru_images_hy_artist_json_packed'
# )