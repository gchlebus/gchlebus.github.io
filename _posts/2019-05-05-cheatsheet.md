---
layout: post
title: "Cheatsheet"
excerpt: ""
date: 2019-05-06
comments: true
---

This is a collection of problems/things that required me to spend quite some time looking for a
solution/answer.


## Jupyter nootebook does not work

### Problem
When editing python files via the jupyter web gui, the python kernel is constantly crashing due to
the following error:
```
ImportError: No module named shutil_get_terminal_size
```
### Solution
Check whether jupyter is installed in the conda environment you are executing the `jupyter notebook`
command. Most probably it is not, so you would need to execute `conda install juypter` in that
environemtn to get rid of the `ImportError`.