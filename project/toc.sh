#!/bin/sh
echo adding frontmatter pages to toc...
cat frontmatter.toc thesis.toc > tmp.toc
mv tmp.toc thesis.toc
echo ...done adding frontmatter pages to toc
