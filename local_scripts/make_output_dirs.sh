#!/usr/bin/env bash


for outer_dir in "outputs/webtext_gpt2" "outputs/webtext_gpt2-large" "outputs/webtext_gpt2-medium" "outputs/webtext_gpt2-xl"
do
for dir in outs save generations metrics
do
    mkdir -p ${outer_dir}/${dir}
    mkdir -p ${outer_dir}/${dir}/ref
    mkdir -p ${outer_dir}/${dir}/basic
done
done


