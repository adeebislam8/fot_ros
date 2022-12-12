#include <iostream>
// #include <vector>
// #include <list>
// #include <cmath>
#include <math.h>
using namespace std;

extern "C"
{
    struct frenet_coordinate {
        double s, d;
    };
    
    struct cartesian_coordiante {
        double x, y, heading;
    };
    // typedef struct frenet_coordinate Struct;

    double dotProduct(double *vect_A, double *vect_B)
    {
    
        double product = 0.0;
        // product = 0.0;
        // Loop for calculate cot product
        for (uint i = 0; i < 2; i++)
        {
            product = product + (vect_A[i] * vect_B[i]);
        }
        return product;
    }

    void crossProduct(double vect_A[], double vect_B[], double cross_P[])
    
    {
    
        cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
        cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
        cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
    }


    // Function that return
    // dot product of two vector array.

    double get_dist(double x, double y, double _x, double _y)
    {
        return sqrt(double(pow((x - _x), 2) + pow((y - _y), 2)));
    }


    int get_closest_waypoints(double x, double y, double *mapx, double *mapy, long length)
    {
        double min_len = 1e10;
        int closest_wp = 0;

        double _mapx, _mapy, dist;

        for (uint i = 0 ; i < length; i++)
        {
            _mapx = mapx[i];
            _mapy = mapy[i];
            dist = get_dist(x,y, _mapx, _mapy);

            if (dist < min_len)
            {
                min_len = dist;
                closest_wp = i;
            }
        }
        return closest_wp;
    }


    int next_waypoint(double x, double y, double *mapx, double *mapy, long length)
    {
        int next_wp;
        int closest_wp = get_closest_waypoints(x, y, mapx, mapy, length);
        // list<double> map_vec = {mapx[closest_wp + 1] - mapx[closest_wp], mapy[closest_wp + 1] - mapy[closest_wp]};
        // list<double> ego_vec = {x - mapx[closest_wp], y - mapy[closest_wp]};  
        // if ((closest_wp+1) >= int(sizeof(mapx)))  
        // {
        //     cout << "sizeof(mapx) " << sizeof(mapx) << " closest_wp + 1 " << closest_wp + 1 << endl;
        // }
        // cout << "sizeof(mapx) " << sizeof(mapx) << " closest_wp + 1 " << closest_wp + 1 << endl;

        double map_vec[] = {mapx[closest_wp + 1] - mapx[closest_wp], mapy[closest_wp + 1] - mapy[closest_wp]};
        double ego_vec[] = {x - mapx[closest_wp], y - mapy[closest_wp]};   

        double direction = dotProduct(map_vec, ego_vec);
        // cout << direction << endl;
        if (direction >= 0.0)
        {
            next_wp = closest_wp + 1;
        }
        else
        {
            next_wp = closest_wp;
        }

        // cout << "closest_wp " << closest_wp << endl;
        // cout << "next_wp " << next_wp << endl;
        return next_wp;
    }


    frenet_coordinate get_frenet(double x, double y, double *mapx, double *mapy, long length)
    {
        // cout << "x " << x << "\ny " << y << "\nmapx[5] " << mapx[5] << "\nmapy[5] " << mapy[5] << "\nlength " << length << endl;

        frenet_coordinate frenet_point;

        // frenet_coordinate *frenet_point;
        // frenet_point = (frenet_coordinate*)malloc(sizeof(frenet_coordinate));
        int next_wp = next_waypoint(x, y, mapx, mapy, length);
        int prev_wp;
        double frenet_s = 0.0;
        // double frenet_d = 0.0;

        // if (next_wp == false)
        // {
        //     frenet_point->s = false;
        //     frenet_point->d = false;
        //     return frenet_point;
        // }

        if ((next_wp - 2) > 0)
        {
            prev_wp = next_wp -2;
        }
        else
        {
            next_wp = next_wp + 2;
            prev_wp = 0;
        }


        double n_x = mapx[next_wp] - mapx[prev_wp];
        double n_y = mapy[next_wp] - mapy[prev_wp];
        double x_x = x - mapx[prev_wp];
        double x_y = y - mapy[prev_wp];


        double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
        double proj_x = proj_norm*n_x;
        double proj_y = proj_norm*n_y;

        double frenet_d = get_dist(x_x,x_y,proj_x,proj_y);


        double ego_vec[] = {x-mapx[prev_wp], y-mapy[prev_wp], 0};
        double map_vec[] = {n_x, n_y, 0};



        double d_cross[3];
        crossProduct(ego_vec, map_vec, d_cross);

        if (d_cross[sizeof(d_cross)/sizeof(d_cross[0])-1] > 0)
        {
            // cout << "d_cross " << d_cross[0] << " " << d_cross[1] << " " << d_cross[2] << endl;
            frenet_d = -frenet_d;
        }

        for (int i = 0; i < prev_wp; i++)
        {
            frenet_s = frenet_s + get_dist(mapx[i],mapy[i],mapx[i+1],mapy[i+1]);
        }

        frenet_s = frenet_s + get_dist(0,0,proj_x,proj_y);

        frenet_point.s = frenet_s;
        frenet_point.d = frenet_d;

        // cout << "frenet_s: " << frenet_point.s << " frenet_d: " << frenet_point.d << endl;
        return frenet_point;
    }

    cartesian_coordiante get_cartesian(double s, double d, double *mapx, double *mapy, double *mapz, long len_wp_s)
    {
        // cout << "s " << s << "\nd " << d << "\nmapx[5] " << mapx[5] << "\nmapy[5] " << mapy[5]  << "\nlength " << len_wp_s << "\nwp_s[5] " << mapz[5] << endl;

        int prev_wp = 0;
        s = fmod(s, mapz[len_wp_s - 1]);
        while ((s > mapz[prev_wp+1]) && (prev_wp < len_wp_s - 2)) 
        {
            prev_wp = prev_wp + 1;
        }

        int next_wp = fmod(prev_wp + 1, len_wp_s);
        double dx = mapx[next_wp] - mapx[prev_wp];
        double dy = mapy[next_wp] - mapy[prev_wp];

        double heading = atan2(dy,dx);
        double seg_s = s - mapz[prev_wp];

        double seg_x = mapx[prev_wp] + seg_s*cos(heading);
        double seg_y = mapy[prev_wp] + seg_s*sin(heading);

        double perp_heading = heading + 90 * M_PI/180;
        double x = seg_x + d*cos(perp_heading);
        double y = seg_y + d*sin(perp_heading);

        cartesian_coordiante cartesian_point;
        cartesian_point.x = x;
        cartesian_point.y = y;
        cartesian_point.heading = heading;

        return cartesian_point;

  
    }
    
    void free_pointer(frenet_coordinate *p)
    {
        free(p);
    }

}

int main()
{
    double x = 0.4;
    double y = 3.0;
    double mapx[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.8, 0.9, 0.10, 0.11, .132};
    double mapy[] = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};

    frenet_coordinate check_point =  get_frenet(x, y, mapx, mapy, sizeof(mapx)/sizeof(mapx[0]));
    cout << "check point3: " << check_point.s << "\n" << check_point.d << endl;
    
    double s = 3.5;
    double d = 0.4;

    // double maps[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.8, 0.9, 0.10, 0.11, 0.132};
    // double mapd[] = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    double wp_s[] = {0.0, 0.5, 0.6, 1.3, 1.6, 1.9, 2.4, 2.9, 3.0, 3.4, 3.7, 3.9, 4.6};
    cartesian_coordiante cart_point = get_cartesian(s,d,mapx,mapy,wp_s, sizeof(wp_s)/sizeof(wp_s[0]));
    cout << "cart point: " << cart_point.x << "\n" << cart_point.y << "\n" << cart_point.heading << endl;
    // free_pointer(check_point);

    return 0;
}