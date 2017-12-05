/*****************************************************************************
**   PL-SLAM: stereo visual SLAM with points and line segment features  	**
******************************************************************************
**																			**
**	Copyright(c) 2017, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2017, MAPIR group, University of Malaga					**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#include <stereoFrame.h>
#include <stereoFrameHandler.h>
#include <ctime>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

#include <mapFeatures.h>
#include <mapHandler.h>

using namespace StVO;
using namespace PLSLAM;

int main(int argc, char **argv)
{
    // read dataset root dir fron environment variable
    string dataset_dir = "/home/kevin/Downloads/kitti/2011_09_26/2011_09_26_drive_0001_extract/";

    // read content of the .yaml dataset configuration file
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+"/kitti.yaml");

    // setup camera
    YAML::Node cam_config = dset_config["cam0"];
    string camera_model = cam_config["cam_model"].as<string>();
    PinholeStereoCamera*  cam_pin;
    bool rectify = false;
    if( camera_model == "Pinhole" )
    {
        // if EuRoC or Falcon yaml file
        if( cam_config["Kl"].IsDefined() )
        {
            rectify = true;
            Mat Kl, Kr, Dl, Dr, R, t;
            vector<double> Kl_ = cam_config["Kl"].as<vector<double>>();
            vector<double> Kr_ = cam_config["Kr"].as<vector<double>>();
            vector<double> Dl_ = cam_config["Dl"].as<vector<double>>();
            vector<double> Dr_ = cam_config["Dr"].as<vector<double>>();
            Kl = ( Mat_<float>(3,3) << Kl_[0], 0.0, Kl_[2], 0.0, Kl_[1], Kl_[3], 0.0, 0.0, 1.0 );
            Kr = ( Mat_<float>(3,3) << Kr_[0], 0.0, Kr_[2], 0.0, Kr_[1], Kr_[3], 0.0, 0.0, 1.0 );
            // load rotation and translation
            vector<double> R_ = cam_config["R"].as<vector<double>>();
            vector<double> t_ = cam_config["t"].as<vector<double>>();
            R = Mat::eye(3,3,CV_64F);
            t = Mat::eye(3,1,CV_64F);
            int k = 0;
            for( int i = 0; i < 3; i++ )
            {
                t.at<double>(i,0) = t_[i];
                for( int j = 0; j < 3; j++, k++ )
                    R.at<double>(i,j) = R_[k];
            }
            // load distortion parameters
            int Nd = Dl_.size();
            Dl = Mat::eye(1,Nd,CV_64F);
            Dr = Mat::eye(1,Nd,CV_64F);
            for( int i = 0; i < Nd; i++ )
            {
                Dl.at<double>(0,i) = Dl_[i];
                Dr.at<double>(0,i) = Dr_[i];
            }
            // if dtype is equidistant (now it is default)
            if( cam_config["dtype"].IsDefined() )
            {
                cam_pin = new PinholeStereoCamera(
                    cam_config["cam_width"].as<double>(),
                    cam_config["cam_height"].as<double>(),
                    cam_config["cam_bl"].as<double>(),
                    Kl, Kr, R, t, Dl, Dr, true);

            }
            else
            // create camera object for EuRoC
                cam_pin = new PinholeStereoCamera(
                    cam_config["cam_width"].as<double>(),
                    cam_config["cam_height"].as<double>(),
                    cam_config["cam_bl"].as<double>(),
                    Kl, Kr, R, t, Dl, Dr,false);
        }
        // else
        else
            cam_pin = new PinholeStereoCamera(
                cam_config["cam_width"].as<double>(),
                cam_config["cam_height"].as<double>(),
                fabs(cam_config["cam_fx"].as<double>()),
                fabs(cam_config["cam_fy"].as<double>()),
                cam_config["cam_cx"].as<double>(),
                cam_config["cam_cy"].as<double>(),
                cam_config["cam_bl"].as<double>(),
                cam_config["cam_d0"].as<double>(),
                cam_config["cam_d1"].as<double>(),
                cam_config["cam_d2"].as<double>(),
                cam_config["cam_d3"].as<double>()  );
    }
    else
    {
        cout << endl << "Not implemented yet." << endl;
        return -1;
    }

    // setup image directories
    string img_dir_l = dataset_dir + "image_00/data/";
    string img_dir_r = dataset_dir + "image_01/data/";

    // get a sorted list of files in the img directories
    boost::filesystem::path img_dir_path_l(img_dir_l.c_str());
    if (!boost::filesystem::exists(img_dir_path_l))
    {
        cout << endl << "Left image directory does not exist: \t" << img_dir_l << endl;
        return -1;
    }
    boost::filesystem::path img_dir_path_r(img_dir_r.c_str());
    if (!boost::filesystem::exists(img_dir_path_r))
    {
        cout << endl << "Right image directory does not exist: \t" << img_dir_r << endl;
        return -1;
    }

    // get all files in the img directories
    size_t max_len_l = 0;
    std::list<std::string> imgs_l;
    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator file(img_dir_path_l); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".pnm"  ||
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs_l.push_back(filename);
            max_len_l = max(max_len_l, filename.length());
        }
    }
    size_t max_len_r = 0;
    std::list<std::string> imgs_r;
    for (boost::filesystem::directory_iterator file(img_dir_path_r); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".pnm"  ||
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs_r.push_back(filename);
            max_len_r = max(max_len_r, filename.length());
        }
    }

    // sort them by filename; add leading zeros to make filename-lengths equal if needed
    std::map<std::string, std::string> sorted_imgs_l, sorted_imgs_r;
    int n_imgs_l = 0, n_imgs_r = 0;
    for (std::list<std::string>::iterator img = imgs_l.begin(); img != imgs_l.end(); ++img)
    {
        sorted_imgs_l[std::string(max_len_l - img->length(), '0') + (*img)] = *img;
        n_imgs_l++;
    }
    for (std::list<std::string>::iterator img = imgs_r.begin(); img != imgs_r.end(); ++img)
    {
        sorted_imgs_r[std::string(max_len_r - img->length(), '0') + (*img)] = *img;
        n_imgs_r++;
    }
    if( n_imgs_l != n_imgs_r)
    {
        cout << endl << "Different number of left and right images." << endl;
        return -1;
    }

    // create PLSLAM object
    PLSLAM::MapHandler* map = new PLSLAM::MapHandler(cam_pin);

    // initialize and run PL-StVO
    int frame_counter = 0;
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);

    for (std::map<std::string, std::string>::iterator it_l = sorted_imgs_l.begin(), it_r = sorted_imgs_r.begin();
         it_l != sorted_imgs_l.end(), it_r != sorted_imgs_r.end(); ++it_l, ++it_r, frame_counter++)
    {
        // load images
        boost::filesystem::path img_path_l = img_dir_path_l / boost::filesystem::path(it_l->second.c_str());
        boost::filesystem::path img_path_r = img_dir_path_r / boost::filesystem::path(it_r->second.c_str());
        Mat img_l( imread(img_path_l.string(), CV_LOAD_IMAGE_UNCHANGED) );  assert(!img_l.empty());
        Mat img_r( imread(img_path_r.string(), CV_LOAD_IMAGE_UNCHANGED) );  assert(!img_r.empty());

        // if images are distorted
        if( rectify )
        {
            Mat img_l_rec, img_r_rec;
            cam_pin->rectifyImagesLR(img_l,img_l_rec,img_r,img_r_rec);
            img_l = img_l_rec;
            img_r = img_r_rec;
        }

        // initialize
        if( frame_counter == 0 ) {
            StVO->initialize(img_l,img_r,0);
            PLSLAM::KeyFrame* kf = new PLSLAM::KeyFrame( StVO->prev_frame, 0 );
            map->initialize( kf );
            // scene.initViewports( img_l.cols, img_r.rows );
        } else { // run
            // PL-StVO
            StVO->insertStereoPair( img_l, img_r, frame_counter );
            StVO->optimizePose();
            cout << "Frame #" << frame_counter << endl;
            // check if a new keyframe is needed
            if( StVO->needNewKF() ) {
                cout <<         "#KeyFrame:     " << map->max_kf_idx + 1;
                cout << endl << "#Points:       " << map->map_points.size();
                cout << endl << "#Lines:     " << map->map_lines.size();
                // grab StF and update KF in StVO (the StVO thread can continue after this point)
                PLSLAM::KeyFrame* curr_kf = new PLSLAM::KeyFrame( StVO->curr_frame );
                // update KF in StVO
                StVO->currFrameIsKF();
                // add keyframe and features to map
                map->addKeyFrame( curr_kf );

                Mat img_frame = StVO->curr_frame->plotStereoFrame();
                imshow("StereoFrame", img_frame);

                // Mat img_match = StVO->curr_frame->plotStereoMatches();
                // imshow("StereoMatch", img_match);

                waitKey(1);
            }
            // update StVO
            StVO->updateFrame();
        }
    }

    // finish SLAM
    map->finishSLAM();

    // perform GBA
    cout << endl << "Performing Global Bundle Adjustment..." ;
    map->globalBundleAdjustment();
    cout << " ... done." << endl;

    return 0;

}

