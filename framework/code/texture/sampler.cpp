// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include <cassert>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include "texture.hpp"
#include "nlohmann/json.hpp"

using Json = nlohmann::json;


const static std::map<std::string, SamplerAddressMode> cSamplerAddressModeByName{
    {"Undefined",  SamplerAddressMode::Undefined},
    {"Repeat", SamplerAddressMode::Repeat},
    {"MirroredRepeat", SamplerAddressMode::MirroredRepeat},
    {"ClampEdge", SamplerAddressMode::ClampEdge},
    {"ClampBorder", SamplerAddressMode::ClampBorder},
    {"MirroredClampEdge", SamplerAddressMode::MirroredClampEdge}
};

const static std::map<std::string, SamplerFilter> cSamplerFilterByName{
    {"Undefined",  SamplerFilter::Undefined},
    {"Nearest", SamplerFilter::Nearest},
    {"Linear", SamplerFilter::Linear}
};

static void from_json( const Json& j, SamplerAddressMode& op ) {
    const auto foundIt = cSamplerAddressModeByName.find( j );
    if (foundIt != cSamplerAddressModeByName.end()) {
        op = foundIt->second;
    }
    else {
        throw std::invalid_argument( "Unknown SamplerAddressMode" );
    }
}

static void from_json( const Json& j, SamplerFilter& op ) {
    const auto foundIt = cSamplerFilterByName.find( j );
    if (foundIt != cSamplerFilterByName.end()) {
        op = foundIt->second;
    }
    else {
        throw std::invalid_argument( "Unknown SamplerFilter" );
    }
}

void from_json( const Json& j, CreateSamplerObjectInfo& info ) {
    auto it = j.find( "Mode" );
    if (it != j.end()) it->get_to( info.Mode );

    it = j.find( "Filter" );
    if (it != j.end()) it->get_to( info.Filter );

    it = j.find( "MipFilter" );
    if (it != j.end()) it->get_to( info.MipFilter );

    it = j.find( "MipBias" );
    if (it != j.end()) it->get_to( info.MipBias);

    it = j.find( "Anisotropy" );
    if (it != j.end()) it->get_to( info.Anisotropy );

    it = j.find( "MinLod" );
    if (it != j.end()) it->get_to( info.MinLod );

    it = j.find( "MaxLod" );
    if (it != j.end()) it->get_to( info.MaxLod );
}
