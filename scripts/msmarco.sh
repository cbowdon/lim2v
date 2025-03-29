#!/usr/bin/env bash

# from https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md

archive_dir=collections/msmarco-passage
msmarco_archive="${archive_dir}/collectionsandqueries.tar.gz"

mkdir -p "$archive_dir"

if [ ! -f "$msmarco_archive" ]; then
    curl -o "$msmarco_archive" https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz;
fi

# Alternative mirror:
# wget https://www.dropbox.com/s/9f54jg2f71ray3b/collectionandqueries.tar.gz -P collections/msmarco-passage

tar xvfz "$msmarco_archive" -C "$archive_dir"