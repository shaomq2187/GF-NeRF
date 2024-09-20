//
// Created by ppwang on 2022/9/16.
//
#pragma once
#include <torch/torch.h>
#include "../Utils/Pipe.h"


class Field : public Pipe {
using Tensor = torch::Tensor;

public:
  virtual Tensor Query(const Tensor& coords) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor Query(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor AnchoredQuery(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }
  virtual std::vector<Tensor> GetParams()
  {    
    CHECK(false) << "Not implemented";
    std::vector<Tensor> ret;
    return ret;
  }
  virtual void SetFeatPoolRequireGrad(bool require_grad)
  {
    CHECK(false) << "Not implemented";
  }
  virtual void ReleaseResources()
  {
    CHECK(false) << "Not implemented";
  }
  virtual void to(std::string device)
  {
    CHECK(false) << "Not implemented";
  }

};
