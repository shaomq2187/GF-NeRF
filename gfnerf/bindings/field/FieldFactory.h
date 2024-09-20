//
// Created by ppwang on 2022/9/16.
//

#pragma once
#include <memory>
#include "Field.h"

std::unique_ptr<Field> ConstructField(int log2_table_size,int mlp_hidden_dim,int mlp_out_dim, int n_hidden_layers,int n_volume,float learn_rate);