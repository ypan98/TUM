#include <iostream>
#include <fstream>
#include <array>

#include "Eigen.h"
#include "VirtualSensor.h"
#include <direct.h>

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

// returns true for valid triangle (not MINF) and with edge length < th
bool validTriangle(Vertex v1, Vertex v2, Vertex v3, float th) {
	// check no vectex is MINF (checking x is already enough)
	if (v1.position.x() == MINF || v2.position.x() == MINF || v3.position.x() == MINF) return false;
	// check no edge length greater than th
	if (((v1.position - v2.position).norm() >= th) ||
		((v1.position - v3.position).norm() >= th) ||
		((v2.position - v3.position).norm() >= th)) return false;
	return true;
}

float zeroWhenInvalid(float f) {
	return f == MINF ? 0.0: f;
}

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = width*height;

	// TODO: Determine number of valid faces
	unsigned nFaces = 0;
	Vector3i* face_idxs = new Vector3i[width*height*2]; // maximum 2*w*h faces
 	// move 2-2 (cell)
	for (unsigned i = 0; i < height-1; i++) {
		for (unsigned j = 0; j < width-1; j++) {
			unsigned idx1 = i * width + j;
			unsigned idx2 = i * width + j + 1;
			unsigned idx3 = (i + 1) * width + j;
			unsigned idx4 = (i + 1) * width + j + 1;
			// upper triangle
			if (validTriangle(vertices[idx1], vertices[idx3], vertices[idx2], edgeThreshold)) {
				face_idxs[nFaces] = Vector3i(idx1, idx3, idx2);
				nFaces++;
			}
			// lower triangle
			if (validTriangle(vertices[idx3], vertices[idx4], vertices[idx2], edgeThreshold)) {
				face_idxs[nFaces] = Vector3i(idx3, idx4, idx2);
				nFaces++;
			}
		}
	}


	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;

	outFile << "# numVertices numFaces numEdges" << std::endl;

	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	for (unsigned i = 0; i < width * height; i++) {
		Vertex v = vertices[i];
		outFile << zeroWhenInvalid(v.position.x()) << " " << zeroWhenInvalid(v.position.y()) << " " << zeroWhenInvalid(v.position.z()) << " ";
		outFile << (unsigned)v.color.x() << " " << (unsigned)v.color.y() << " " << (unsigned)v.color.z() << " " << (unsigned)v.color[3] << "\n";
	}


	// TODO: save valid faces
	std::cout << "# list of faces" << std::endl;
	std::cout << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
	for (unsigned i = 0; i < nFaces; i++) {
		Vector3i idxs = face_idxs[i];
		outFile << "3 " << idxs.x() << " " << idxs.y() << " " << idxs.z() << "\n";
	}


	// close file
	outFile.close();

	return true;
}

int main()
{
	// Make sure this path points to the data folder
	std::string filenameIn = "D:\\TUM\\3DScan\\Data\\rgbd_dataset_freiburg1_xyz\\";
	std::string filenameBaseOut = "mesh_";

	// load video
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

		float fX = depthIntrinsics(0, 0);
		float fY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);

		// compute inverse depth extrinsics
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		Vertex* vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];
		for (unsigned y = 0; y < sensor.GetDepthImageHeight(); y++) {
			for (unsigned x = 0; x < sensor.GetDepthImageWidth(); x++) {
				unsigned int idx = y * sensor.GetDepthImageWidth() + x;
				float z = depthMap[idx];
				if (z == MINF) {
					vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
					vertices[idx].color = Vector4uc(0, 0, 0, 0);
				}
				else {
					// dehomogenization and pixel to camera space of depth camera
					Vector3f v_c_depth_camera = depthIntrinsicsInv * Vector3f(x * z, y * z, z);
					// to camera space of sensor camera
					Vector4f v_c = depthExtrinsicsInv * Vector4f(v_c_depth_camera[0], v_c_depth_camera[1], v_c_depth_camera[2], 1.0);
					// to world space
					vertices[idx].position = trajectoryInv * v_c;
					// color map
					unsigned int colormap_idx = idx * 4;
					vertices[idx].color = Vector4uc(colorMap[colormap_idx], colorMap[colormap_idx + 1], colorMap[colormap_idx + 2], colorMap[colormap_idx + 3]);
				}
			}
		}

		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}