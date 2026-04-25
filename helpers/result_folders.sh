#!/bin/bash

ls -la | grep "results" | awk '{print $9}' |  paste -s -d,
