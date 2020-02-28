/*
 * @file CountingBees.h
 */

 
#ifndef COUNTING_BEES_H
#define COUNTING_BEES_H


// ------------ Include Files -------------------------------------------------
#include <vector>


// ------------ defines -------------------------------------------------------



// ------------ class defines -------------------------------------------------
class CountingBees
{
protected:
    typedef struct tagTRACK_NODE
    {
        int   nFrame;
        int   nID;
        float fPosX;
        float fPosY;
        float fWidth;
        float fHeight;
    } TRACK_NODE, *PTRACK_NODE;

    typedef struct tagTRACK_POINTS
    {
        int   nID;
        float fPosX;
        float fPosY;
    } TRACK_POINT, *PTRACK_POIN;

    typedef std::vector<TRACK_NODE>    vectorNodes;
    typedef std::vector<TRACK_POINT>   vectorPoints;
    typedef std::vector<vectorPoints>  vectorTrackList;

public:
    CountingBees(int xGateLeft, int yGateTop, int xGateRight, int yGateBottom);

public:
    bool Count(int& cnInBees, int& cnOutBees);
        
public:
    void Update(int nFrame, int idTrack, float x, float y, float width, float height);

protected:
    float GetTrackAngle(const TRACK_POINT& ptStart, const TRACK_POINT& ptEnd);
    void  SortVectorInts(const std::vector<int>& rgIntSrc, std::vector<int>& rgIntDest);
    bool  GetMaxMinFrame(int& nFrameMin, int& nFrameMax);
    
protected:
    int m_xGateLeft;
    int m_yGateTop;
    int m_xGateRight;
    int m_yGateBottom;

    std::vector<TRACK_NODE> m_rgTrackNodes;
};


#endif // COUNTING_BEES_H

