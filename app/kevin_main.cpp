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

#ifdef HAS_MRPT
#include <slamScene.h>
#include <mrpt/utils/CTicTac.h>
#endif

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
    // read content of the .yaml dataset configuration file
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+"/dataset_params.yaml");

    // setup camera
    YAML::Node cam_config = dset_config["cam0"];
    string camera_model = cam_config["cam_model"].as<string>();
    PinholeStereoCamera*  cam_pin;
    bool rectify = false;
    if( camera_model == "Pinhole" ) {
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

    // create scene
    #ifdef HAS_MRPT
    string scene_cfg_name;
    if( (dataset_name.find("kitti")!=std::string::npos) ||
        (dataset_name.find("malaga")!=std::string::npos)  )
        scene_cfg_name = "../config/scene_config.ini";
    else
        scene_cfg_name = "../config/scene_config_indoor.ini";
    slamScene scene(scene_cfg_name);
    Matrix4d Tcw, Tfw = Matrix4d::Identity(), Tfw_prev = Matrix4d::Identity(), T_inc;
    Vector6d cov_eig;
    Matrix6d cov;
    Tcw = Matrix4d::Identity();
    scene.setStereoCalibration( cam_pin->getK(), cam_pin->getB() );
    scene.initializeScene(Tfw);
    #endif

    // create PLSLAM object
    PLSLAM::MapHandler* map = new PLSLAM::MapHandler(cam_pin);

    // initialize and run PL-StVO
    int frame_counter = 0;
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);
    for (std::map<std::string, std::string>::iterator it_l = sorted_imgs_l.begin(), it_r = sorted_imgs_r.begin();
         it_l != sorted_imgs_l.end(), it_r != sorted_imgs_r.end(); ++it_l, ++it_r, frame_counter++)
    {

        // load images
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
        if( frame_counter == 0 )
        {
            StVO->initialize(img_l,img_r,0);
            PLSLAM::KeyFrame* kf = new PLSLAM::KeyFrame( StVO->prev_frame, 0 );
            map->initialize( kf );
            scene.initViewports( img_l.cols, img_r.rows );
        }
        // run
        else
        {
            // PL-StVO
            StVO->insertStereoPair( img_l, img_r, frame_counter );
            StVO->optimizePose();
            cout << "------------------------------------------   Frame #" << frame_counter
                 << "   ----------------------------------------" << endl;
            // check if a new keyframe is needed
            if( StVO->needNewKF() )
            {
                cout <<         "#KeyFrame:     " << map->max_kf_idx + 1;
                cout << endl << "#Points:       " << map->map_points.size();
                cout << endl << "#Segments:     " << map->map_lines.size();
                cout << endl << endl;
                // grab StF and update KF in StVO (the StVO thread can continue after this point)
                PLSLAM::KeyFrame* curr_kf = new PLSLAM::KeyFrame( StVO->curr_frame );
                // update KF in StVO
                StVO->currFrameIsKF();
                // add keyframe and features to map
                map->addKeyFrame( curr_kf );
                // update scene
                #ifdef HAS_MRPT
                imwrite("../config/aux/img_aux.png",StVO->curr_frame->plotStereoFrame());
                scene.setImage( "../config/aux/img_aux.png" );
                scene.updateScene( map );
                #endif
            }
            // update StVO
            StVO->updateFrame();

        }
    }

    // finish SLAM
    map->finishSLAM();
    #ifdef HAS_MRPT
    scene.setImage( "../config/aux/img_aux.png" );
    scene.updateSceneGraphs( map );
    #endif

    getchar();

    // perform GBA
    cout << endl << "Performing Global Bundle Adjustment..." ;
    map->globalBundleAdjustment();
    cout << " ... done." << endl;
    #ifdef HAS_MRPT
    scene.setImage( "../config/aux/img_aux.png" );
    scene.updateSceneGraphs( map );
    #endif

    // wait until the scene is closed
    #ifdef HAS_MRPT
    while( scene.isOpen() );
    #endif

    return 0;

}
