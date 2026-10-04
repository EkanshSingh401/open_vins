/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// ROS 2 counterpart of ros1_serial_msckf.cpp: read a rosbag2 bag directly and
// feed it to the estimator in recorded order, on ONE thread, as fast as it can.
//
// Purpose: deterministic offline evaluation. Replaying a bag into
// run_subscribe_msckf is not repeatable -- that node forces
// use_multi_threading_subs and spins a MultiThreadedExecutor, so which IMU
// samples are buffered when an image is processed depends on DDS delivery
// timing, and two replays of one bag diverge from the first state onwards.
// Here the order of every callback is fixed by the bag.
//
// It drives the SAME ROS2Visualizer callbacks as the live node
// (callback_inertial / callback_stereo), so every output is produced by the
// same code path: published topics including /openvins/joint_covariance, and
// the save_total_state files. run_subscribe_msckf remains the live node; this
// binary is for evaluation only.
//
// Parameters (ROS 2, e.g. -p path_bag:=/data/flight.bag):
//   config_path   estimator YAML (as for run_subscribe_msckf)
//   path_bag      rosbag2 directory (sqlite3 or mcap; FILE-compressed is supported)
//   topic_imu / topic_camera0 / topic_camera1   optional overrides of the
//                 kalibr "rostopic" entries, exactly as in the live node
// Builds on Humble and Jazzy.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "ros/ROS2Visualizer.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_compression/sequential_compression_reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/metadata_io.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

using namespace ov_msckf;

namespace {

// rosbag2 renamed SerializedBagMessage::time_stamp to recv_timestamp after
// Humble. Pick whichever member exists at compile time rather than branching on
// the distro: the int/long overloads make the recv_timestamp form preferred
// when both would be viable.
template <class M> auto bag_time_ns(const M &m, int) -> decltype(m.recv_timestamp) { return m.recv_timestamp; }
template <class M> auto bag_time_ns(const M &m, long) -> decltype(m.time_stamp) { return m.time_stamp; }

double stamp_sec(const builtin_interfaces::msg::Time &t) { return t.sec + t.nanosec * 1e-9; }

// The node auto-declares every -p override, so declaring again would throw.
std::string string_param(const rclcpp::Node::SharedPtr &node, const std::string &name, const std::string &def) {
  if (!node->has_parameter(name))
    node->declare_parameter<std::string>(name, def);
  std::string v;
  node->get_parameter(name, v);
  return v;
}

template <class T> std::shared_ptr<T> deserialize(const rosbag2_storage::SerializedBagMessage &bm) {
  static rclcpp::Serialization<T> ser;
  rclcpp::SerializedMessage sm(*bm.serialized_data);
  auto out = std::make_shared<T>();
  ser.deserialize_message(&sm, out.get());
  return out;
}

} // namespace

int main(int argc, char **argv) {

  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("ros2_serial_msckf", options);
  node->get_parameter<std::string>("config_path", config_path);

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node(node);

  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Repeatability: everything that can run concurrently is forced serial.
  // These are the settings ros1_serial_msckf.cpp marks "uncomment if you want
  // repeatability"; here repeatability is the whole point, so they are on.
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = false;
  params.use_multi_threading_pubs = false;
  params.num_opencv_threads = 0;
  auto sys = std::make_shared<VioManager>(params);
  auto viz = std::make_shared<ROS2Visualizer>(node, sys);

  // Topics: same resolution as ROS2Visualizer::setup_subscribers (a ROS
  // parameter, overridden by the kalibr "rostopic" entry), without subscribing.
  std::string topic_imu = string_param(node, "topic_imu", "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  std::vector<std::string> topic_cameras;
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    std::string cam_topic = string_param(node, "topic_camera" + std::to_string(i), "/cam" + std::to_string(i) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
    topic_cameras.push_back(cam_topic);
  }

  std::string path_bag = string_param(node, "path_bag", "");

  if (!parser->successful() || path_bag.empty()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters (and path_bag must be set)\n" RESET);
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  if (params.state_options.num_cameras != 2) {
    PRINT_ERROR(RED "[SERIAL]: only stereo (2 cameras) is supported, config has %d\n" RESET, params.state_options.num_cameras);
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  PRINT_INFO("[SERIAL]: bag %s\n", path_bag.c_str());
  PRINT_INFO("[SERIAL]: imu %s | cam0 %s | cam1 %s\n", topic_imu.c_str(), topic_cameras.at(0).c_str(), topic_cameras.at(1).c_str());

  // Open the bag. FILE-compressed bags need the compression reader; the plain
  // reader would fail on the zstd frame header.
  rosbag2_storage::MetadataIo metadata_io;
  auto metadata = metadata_io.read_metadata(path_bag);
  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> reader;
  if (metadata.compression_mode == "FILE" || metadata.compression_mode == "file") {
    reader = std::make_unique<rosbag2_compression::SequentialCompressionReader>();
    PRINT_INFO("[SERIAL]: bag is FILE-compressed (%s)\n", metadata.compression_format.c_str());
  } else {
    reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
  }
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = path_bag;
  storage_options.storage_id = metadata.storage_identifier;
  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";
  reader->open(storage_options, converter_options);

  rosbag2_storage::StorageFilter filter;
  filter.topics = {topic_imu, topic_cameras.at(0), topic_cameras.at(1)};
  reader->set_filter(filter);

  // Stream in recorded order. IMU goes straight in. A camera image waits for
  // its partner from the other camera (header stamps within 0.02 s, the same
  // tolerance ros1_serial_msckf.cpp uses); a pair is handed over as soon as it
  // is complete, and an image that never finds a partner is dropped once it is
  // 0.5 s stale.
  std::map<int, std::vector<sensor_msgs::msg::Image::ConstSharedPtr>> pending;
  size_t n_imu = 0, n_img[2] = {0, 0}, n_pairs = 0, n_unpaired = 0;
  int64_t prev_bag_ns = 0;
  size_t out_of_order = 0;
  while (rclcpp::ok() && reader->has_next()) {
    auto bm = reader->read_next();
    int64_t t_bag = static_cast<int64_t>(bag_time_ns(*bm, 0));
    if (t_bag < prev_bag_ns)
      out_of_order++;
    prev_bag_ns = t_bag;

    if (bm->topic_name == topic_imu) {
      viz->callback_inertial(deserialize<sensor_msgs::msg::Imu>(*bm));
      n_imu++;
      continue;
    }
    int cam = (bm->topic_name == topic_cameras.at(0)) ? 0 : (bm->topic_name == topic_cameras.at(1)) ? 1 : -1;
    if (cam < 0)
      continue;
    sensor_msgs::msg::Image::ConstSharedPtr img = deserialize<sensor_msgs::msg::Image>(*bm);
    n_img[cam]++;
    double t_img = stamp_sec(img->header.stamp);

    int other = 1 - cam;
    auto &cand = pending[other];
    auto match = cand.end();
    for (auto it = cand.begin(); it != cand.end(); ++it) {
      if (std::abs(stamp_sec((*it)->header.stamp) - t_img) < 0.02) {
        match = it;
        break;
      }
    }
    if (match != cand.end()) {
      auto img0 = (cam == 0) ? img : *match;
      auto img1 = (cam == 0) ? *match : img;
      cand.erase(match);
      viz->callback_stereo(img0, img1, 0, 1);
      n_pairs++;
    } else {
      pending[cam].push_back(img);
    }
    for (auto &kv : pending) {
      auto &v = kv.second;
      for (auto it = v.begin(); it != v.end();) {
        if (t_img - stamp_sec((*it)->header.stamp) > 0.5) {
          it = v.erase(it);
          n_unpaired++;
        } else {
          ++it;
        }
      }
    }
  }

  PRINT_INFO("[SERIAL]: done. imu=%zu cam0=%zu cam1=%zu stereo_pairs=%zu unpaired_dropped=%zu out_of_order_bag_stamps=%zu\n",
             n_imu, n_img[0], n_img[1], n_pairs, n_unpaired, out_of_order);
  viz->visualize_final();
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
