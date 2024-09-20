//
// Created by ppwang on 2022/9/16.
//
#include "FieldFactory.h"
#include "Hash3DAnchored.h"

std::unique_ptr<Field> ConstructField(int log2_table_size,int n_volume,float learn_rate) {
  return std::make_unique<Hash3DAnchored>( log2_table_size,n_volume, learn_rate);

}