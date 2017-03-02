// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

float euclideanDistance(float *a, float *b, int n) {
    __m128 sum =  _mm_setzero_ps();
    int i = 0;
    float result = 0;

    for (; i < n - n % 8; i += 8) {
        __m128 sub_result1 = _mm_sub_ps(_mm_load_ps(b +  0 + i), _mm_load_ps(a +  0 + i));
        sum = _mm_add_ps(sum, _mm_mul_ps(sub_result1, sub_result1));

        __m128 sub_result2 = _mm_sub_ps(_mm_load_ps(b +  4 + i), _mm_load_ps(a +  4 + i));
        sum = _mm_add_ps(sum, _mm_mul_ps(sub_result2, sub_result2));
    }

    float sum_array[4] = {0, 0, 0, 0};
    _mm_store_ps(sum_array, sum);

    for (int j = 0; j < 4; j++) {
        result += sum_array[j];
    }

    for (; i < n; i++) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return result;
}

__m128 displacementOptimized(__m128 dxs, __m128 dys) {
    return _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(dxs, dxs), _mm_mul_ps(dys, dys)));
}

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
    const int X_MAX = imageWidth - featureWidth - 4;
    const int Y_MAX = imageHeight - featureHeight;
    // const int MAX_DIFFERENCE = 65280;
    #pragma omp parallel for
    for (int y = 0; y < imageHeight; y++)
    {
        int x = 0;
        for (; x < X_MAX; x+=4)
        {
            /* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
            if ((y < featureHeight) || (y >= Y_MAX) || (x < featureWidth))
            {
                depth[y * imageWidth + x] = 0;
                x -= 3;
                continue;
            }

            __m128 minimumSquaredDifference = _mm_set1_ps(-1);
            __m128 minimumDy = _mm_setzero_ps();
            __m128 minimumDx = _mm_setzero_ps();

            /* Iterate through all feature boxes that fit inside the maximum displacement box.
            centered around the current pixel. */
            for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
            {
                for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
                {
                    /* Skip feature boxes that dont fit in the displacement box. */
                    if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
                    {
                        continue;
                    }

                    __m128 squaredDifference = _mm_setzero_ps();

                    /* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
                    for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
                    {
                        for (int boxX = -featureWidth; boxX <= featureWidth; boxX++)
                        {
                            int leftX = x + boxX;
                            int leftY = y + boxY;
                            int rightX = x + dx + boxX;
                            int rightY = y + dy + boxY;

                            float dump[4];

                            __m128 difference = _mm_sub_ps(_mm_loadu_ps(left + leftY * imageWidth + leftX), _mm_loadu_ps(right + rightY * imageWidth + rightX));

                            squaredDifference = _mm_add_ps(squaredDifference, _mm_mul_ps(difference, difference));

                        }
                    }


                    /*
                    Check if you need to update minimum square difference.
                    This is when either it has not been set yet, the current
                    squared displacement is equal to the min and but the new
                    displacement is less, or the current squared difference
                    is less than the min square difference.
                    */
                    __m128 update = _mm_and_ps(_mm_cmpeq_ps(minimumSquaredDifference, squaredDifference), _mm_cmpgt_ps(displacementOptimized(minimumDx, minimumDy), _mm_set1_ps(displacementNaive(dx, dy))));
                    float dump[4];
                    update = _mm_or_ps(update, _mm_cmpgt_ps(minimumSquaredDifference, squaredDifference));
                    update = _mm_or_ps(update, _mm_cmpeq_ps(minimumSquaredDifference, _mm_set1_ps(-1)));
                    _mm_storeu_ps(dump, _mm_cmpeq_ps(update, _mm_set1_ps(0)));

                    minimumSquaredDifference = _mm_or_ps(_mm_and_ps(squaredDifference, update), _mm_andnot_ps(update, minimumSquaredDifference));
                    _mm_storeu_ps(dump, minimumSquaredDifference);
                    minimumDx = _mm_or_ps(_mm_and_ps(_mm_set1_ps(dx), update), _mm_andnot_ps(update, minimumDx));
                    minimumDy = _mm_or_ps(_mm_and_ps(_mm_set1_ps(dy), update), _mm_andnot_ps(update, minimumDy));

                }
            }



            /*
            Set the value in the depth map.
            If max displacement is equal to 0, the depth value is just 0.
            */

            // printf("done\n");


            __m128 update = _mm_cmpneq_ps(minimumSquaredDifference, _mm_set1_ps(-1));

            if (maximumDisplacement == 0) {
                _mm_storeu_ps(depth + y * imageWidth + x, _mm_setzero_ps());
            } else {
                _mm_storeu_ps(depth + y * imageWidth + x, _mm_and_ps(update, displacementOptimized(minimumDx, minimumDy)));
            }



        }


        for (;x < imageWidth; x++)
        {
            /* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
            if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
            {
                depth[y * imageWidth + x] = 0;
                continue;
            }

            float minimumSquaredDifference = -1;
            int minimumDy = 0;
            int minimumDx = 0;

            /* Iterate through all feature boxes that fit inside the maximum displacement box.
            centered around the current pixel. */
            for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
            {
                for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
                {
                    /* Skip feature boxes that dont fit in the displacement box. */
                    if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
                    {
                        continue;
                    }

                    float squaredDifference = 0;

                    /* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
                    for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
                    {
                        for (int boxX = -featureWidth; boxX <= featureWidth; boxX++)
                        {
                            int leftX = x + boxX;
                            int leftY = y + boxY;
                            int rightX = x + dx + boxX;
                            int rightY = y + dy + boxY;

                            float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
                            squaredDifference += difference * difference;
                        }
                    }

                    /*
                    Check if you need to update minimum square difference.
                    This is when either it has not been set yet, the current
                    squared displacement is equal to the min and but the new
                    displacement is less, or the current squared difference
                    is less than the min square difference.
                    */
                    if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
                    {
                        minimumSquaredDifference = squaredDifference;
                        minimumDx = dx;
                        minimumDy = dy;
                    }
                }
            }

            /*
            Set the value in the depth map.
            If max displacement is equal to 0, the depth value is just 0.
            */
            if (minimumSquaredDifference != -1)
            {
                if (maximumDisplacement == 0)
                {
                    depth[y * imageWidth + x] = 0;
                }
                else
                {
                    depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
                }
            }
            else
            {
                depth[y * imageWidth + x] = 0;
            }
        }
        //
    }
}
