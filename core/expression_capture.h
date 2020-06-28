#pragma once
#include<string>


namespace  expc
{

struct M_Frame {
    int32_t width = -1;
    int32_t height = -1;
    int32_t step = -1;
    int32_t type = -1;
    const unsigned char *data = nullptr;
};
typedef std::pair<M_Frame, M_Frame> M_Frames;


bool init();
void uninit();

bool open_camera(int index = 0);
void close_camera();
void set_source_image(const std::string &filepath);
M_Frame get_image();
M_Frames get_image_predicted();

}
