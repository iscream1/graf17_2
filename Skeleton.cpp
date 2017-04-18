//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Csibi Martin
// Neptun : V5LSRD
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	out vec2 texcoord;			// output attribute: texture coordinate

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";


struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const float& f) {
		vec4 result;
		result.v[0]=f*v[0];
		result.v[1]=f*v[1];
		result.v[2]=f*v[2];
		return result;
	}

	vec4 operator*(const vec4 f) {
		vec4 result;
		result.v[0]=v[1]*f.v[2]-v[2]*f.v[1];
		result.v[1]=v[2]*f.v[0]-v[0]*f.v[2];
		result.v[2]=v[0]*f.v[1]-v[1]-f.v[0];
		return result;
	}

	vec4& operator=(const vec4 f) {
		v[0]=f.v[0];
		v[1]=f.v[1];
		v[2]=f.v[2];
		return *this;
	}

    vec4 operator-(const vec4& vx)
    {
        return vec4(v[0]-vx.v[0], v[1]-vx.v[1], v[2]-vx.v[2]);
    }

    vec4 operator+(const vec4& vx)
    {
        return vec4(v[0]+vx.v[0], v[1]+vx.v[1], v[2]+vx.v[2]);
    }

	vec4 operator+=(const vec4& f) {
		*this=*this+f;
		return *this;
	}

	vec4 operator/(const float& f) {
		vec4 result;
		result.v[0]=v[0]/f;
		result.v[1]=v[1]/f;
		result.v[2]=v[2]/f;
		return result;
	}

	const vec4& normalize()
	{
	    return *this=*this * (1/sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]));
	}

	float length()
	{
	    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	}

	float dot(const vec4 &vx)
	{
	    return v[0]*vx.v[0]+v[1]*vx.v[1]+v[2]*vx.v[2];
	}
};

const float epsilon=1e-4;
#define PI 3.1415
#define DMAX 5
#define maxdepth 5

class Material
{
public:
    vec4 F0;
    float n;
    vec4 ka;
    vec4 kd;
    vec4 ks;
    float shine;
    bool reflective=false;
    bool refractive=false;

    bool isReflective()
    {
        return reflective;
    }

    bool isRefractive()
    {
        return refractive;
    }

    vec4 reflect(vec4 inDir, vec4 normal)
    {
        return inDir-normal*normal.dot(inDir)*2.0f;
    }
    virtual vec4 refract(vec4, vec4)=0;
    virtual vec4 Fresnel(vec4, vec4)=0;
    virtual vec4 shade(vec4, vec4, vec4, vec4)=0;
};

class Ray
{
public:
    vec4 p;
    vec4 dv;

    Ray(vec4 origin=0, vec4 dir=0)
    {
        p=origin;
        dv=dir;
    }
};



class SmoothMaterial : public Material
{
    vec4 F0;
    float n;
    float shine=50;
    vec4 kd=vec4(100,0,50);
    vec4 ks=vec4(0,0,50);
    bool reflective=false;
    bool refractive=false;
public:
    Smoothmaterial()
    {
        kd=vec4(100,0,50);
        ks=vec4(0,0,50);
    }

    vec4 refract(vec4 inDir, vec4 normal)
    {
        float ior=n;
        float cosa=-1*normal.dot(inDir);
        if(cosa<0)
        {
            cosa=-1*cosa;
            normal=normal*-1.0f;
            ior=1/n;
        }
        float disc=1-(1-cosa*cosa)/ior/ior;
        if(disc<0) return reflect(inDir, normal);
        return inDir/ior+normal*(cosa/ior-sqrtf(disc));
    }

    vec4 Fresnel(vec4 inDir, vec4 normal)
    {
        float cosa=fabs(normal.dot(inDir));
        return F0+(vec4(1,1,1)-F0)*pow(1-cosa, 5);
    }

    vec4 shade(vec4 normal, vec4 viewDir, vec4 lightDir, vec4 inRad)
    {
        vec4 reflRad;
        float cosTheta=normal.dot(lightDir);
        if(cosTheta<0) return reflRad;
        reflRad=inRad/255*kd/255*cosTheta;
        vec4 halfway=(viewDir+lightDir).normalize();
        float cosDelta=normal.dot(halfway);
        //printf("%f\n", cosDelta);
        if(cosDelta<0) return reflRad;
        return reflRad+inRad/255*ks/255*pow(cosDelta, shine);
    }
};

struct Hit {
public:
    float t;
    vec4 position;
    vec4 normal;
    Material* material;

    Hit(float tx=-1) { t = tx; }

    void setT(float tx, Ray ray)
    {
        t=tx;
        //printf("asd");
        //material=new SmoothMaterial();
        position=((ray.p)+(ray.dv*t));
    }
};

struct Intersectable
{
      Material* material;
      virtual Hit intersect(Ray& ray)=0;
      virtual vec4 surfNorm(vec4 p)=0;
};

class Camera
{
public:
    int w=windowWidth;
    int h=windowHeight;
    int w2=w/2;
    int h2=h/2;
    vec4 eye;
    vec4 lookat;
    vec4 up;
    vec4 right;
    int XM;
    int YM;

    Ray getRay(int right, int up)
    {
        Ray ray;
        ray.p=vec4(right-w2, up-h2, 0);
        ray.dv=ray.p-eye;
        ray.dv.normalize();
        return ray;
    }
};

Camera camera;

class RoughMaterial : public Material
{
    vec4 kd, ks;
    float shine;
public:
    vec4 shade(vec4 normal, vec4 viewDir, vec4 lightDir, vec4 inRad)
    {
        vec4 reflRad;
        float cosTheta=normal.dot(lightDir);
        if(cosTheta<0) return reflRad;
        reflRad=inRad*kd*cosTheta;
        vec4 halfway=(viewDir+lightDir).normalize();
        float cosDelta=normal.dot(halfway);
        if(cosDelta<0) return reflRad;
        return reflRad+inRad*ks*pow(cosDelta, shine);
    }
};

class Sphere : public Intersectable
{
      vec4 center=vec4(100,100,200);
      float R=100.0f;
public:
    Hit intersect(Ray& ray)
    {
        /*float dirx=ray.dv.v[0];
        float diry=ray.dv.v[1];
        float dirz=ray.dv.v[2];
        float px=ray.p.v[0];
        float py=ray.p.v[1];
        float pz=ray.p.v[2];
        float ox=center.v[0];
        float oy=center.v[1];
        float oz=center.v[2];

        float a=dirx*dirx+diry*diry+dirz*dirz;
        float b=2*dirx*(px-ox)+2*diry*(py-oy)+2*dirz*(pz-oz);
        float c=ox*ox+oy*oy+oz*oz+px*px+py*py+pz*pz-2*(ox*px+oy*py+oz*pz)-R*R;*/

        vec4 dv=ray.dv;
        float a=dv.dot(ray.dv);
        float b=2*(ray.p-center).dot(ray.dv);
        float c=((ray.p-center).dot(camera.eye-center)-R*R);

        float det=b*b-4*a*c;

        //printf("%lf\n", det);

        if(det<(0.0f-epsilon))
        {
            return Hit();
        }

        else if(det<epsilon)
        {
            det=0.0f;
        }

        float x=((-1.0f*b-sqrtf(det)) / (2.0f*a));

        if(x>epsilon)
        {
            Hit ret;
            ret.setT(x, ray);
            ret.normal=surfNorm(ret.position);
            return ret;
        }
        else return Hit(0.0f);
    }

    vec4 surfNorm(vec4 p)
    {
        return (p-center).normalize();
    }
};

class Mesh : public Intersectable
{
    Hit intersect(const Ray& ray);
};


class Color
{
public:
    float r, g, b;

    Color(float rr=0, float gg=0, float bb=0)
    {
        r=rr;
        g=gg;
        b=bb;
    }

    Color(const Color& c)
    {
        r=c.r;
        g=c.g;
        b=c.b;
    }

    Color& operator=(Color c)
    {
        r=c.r;
        g=c.g;
        b=c.b;
        return *this;
    }

    Color operator+(Color& c)
    {
        Color res;
        res.r=r+c.r;
        res.g=g+c.g;
        res.b=b+c.b;
        return res;
    }

    Color operator/(float d)
    {
        Color res;
        res.r=r/d;
        res.g=g/d;
        res.b=b/d;
        return res;
    }
};

class Light
{
public:
    Color color;
    vec4 o=vec4(200, -200, 100);
    vec4 Lout=vec4(255, 128, 60);
    bool type;

    vec4 getLightDir();
    vec4 getInRad();
    vec4 getDist();
};

class Shape
{
public:
    Color Kr;
    Color color;
    Color fr;
    Color kappa;
    bool isReflective;
    bool isRefractive;

    virtual float intersect(Ray& ray)=0;

    virtual vec4 surfNorm(vec4& intersection)=0;

    void ComputerFresnel(float cost)
    {
        Kr.r = ((pow((fr.r - 1.0), 2)) + (pow(kappa.r, 2)) + (pow((1.0 - cost), 5)) * (4 * fr.r)) / ((pow((fr.r + 1.0), 2)) + (pow(kappa.r, 2)));
        Kr.g = ((pow((fr.g - 1.0), 2)) + (pow(kappa.g, 2)) + (pow((1.0 - cost), 5)) * (4 * fr.g)) / ((pow((fr.g + 1.0), 2)) + (pow(kappa.g, 2)));
        Kr.b = ((pow((fr.b - 1.0), 2)) + (pow(kappa.b, 2)) + (pow((1.0 - cost), 5)) * (4 * fr.b)) / ((pow((fr.b + 1.0), 2)) + (pow(kappa.b, 2)));
    }
};



vec4** wo;

class Scene
{
public:
    Intersectable* shapes[100];
    int shc=0;
    Light light;
    //int lightc=0;
    vec4 La=vec4(0, 191, 255);
    Camera camera;

    //light.o=new vec4(500, 500, 500, 1);

    Scene()
    {
        light.color=Color(255.0f,128.0f,60.0f);
        light.o=vec4(200.0f,-200.0f,0.0f);
        shc=0;
    }

    void build();

    void render()
    {
        int w=windowWidth;
        int h=windowHeight;

        wo=new vec4*[w];


        for(int i=0;i<w;i++)
            wo[i]=new vec4[h];

        for(int i=0;i<h;i++)
        {
            for(int j=0;j<w;j++)
            {
                Ray ray=camera.getRay(j, i);
                wo[i][j]=trace(ray, 0);
            }
        }
    }

    void add(Intersectable* added)
    {
        shapes[shc]=added;
        shc++;
    }

    Hit firstIntersect(Ray ray)
    {
        Hit bestHit;
        for(int i=0;i<shc;i++)
        {
            Hit hit = shapes[i]->intersect(ray);
            hit.material=shapes[i]->material;
            if(hit.t > 0.0f && (bestHit.t < 0.0f || hit.t < bestHit.t))
            {
                bestHit = hit;
            }
        }
        return bestHit;
    }

    float sign(float a)
    {
        return a>0;
    }


    vec4 trace(Ray ray, int depth)
    {
        if(depth>maxdepth) return La;
        Hit hit=firstIntersect(ray);

        if(hit.t<0) return La;


        vec4 outRadiance=La*hit.material->ka;

        {

            vec4 ip=(ray.p)+(ray.dv*hit.position);
            vec4 normal=hit.normal;
            Ray ir;
            ir.p=ip+normal*0.01f;
            ir.dv=light.o-ip;
            ir.dv.normalize();
            float f=normal.dot(ir.dv);
            if(f<0.0f) f=0.0f;

            //printf("%f ", hit.material->kd.v[0]);
            vec4 c=hit.material->kd;
            //printf("%f ", c.v[0]);
            c.v[0]=c.v[0]*(light.Lout.v[0])*f;
            c.v[1]=c.v[1]*(light.Lout.v[1])*f;
            c.v[2]=c.v[2]*(light.Lout.v[2])*f;

            return c;

            Ray shadowRay(hit.position+hit.normal*epsilon*sign((ray.dv*-1.0f).dot(hit.position)), light.o-hit.position);
            Hit shadowHit=firstIntersect(shadowRay);
            if((shadowHit.t<0.0f)||(shadowHit.t>(hit.position-light.o).length()))
                outRadiance+=hit.material->shade(hit.normal,(ray.dv*-1.0f),light.o-hit.position,light.Lout);
        }
        if(hit.material->isReflective())
        {
            vec4 reflectionDir=hit.material->reflect((ray.dv*-1.0f), hit.normal);
            Ray reflectedRay(hit.position+hit.normal*epsilon*sign((ray.dv*-1.0f).dot(hit.normal)), reflectionDir);
            outRadiance+=trace(reflectedRay, depth+1)*hit.material->Fresnel((ray.dv*-1.0f), hit.normal);
        }
        if(hit.material->isRefractive())
        {
            vec4 refractionDir=hit.material->refract((ray.dv*-1.0f), hit.normal);
            Ray refractedRay(hit.position-hit.normal*epsilon*sign((ray.dv*-1.0f).dot(hit.normal)), refractionDir);
            outRadiance+=trace(refractedRay, depth+1)*(vec4(1,1,1)-hit.material->Fresnel((ray.dv*-1.0f), hit.normal));
        }
        return outRadiance;
    }

    void Create()
    {

    }
};

class cShape : public Shape
{
public:
    vec4 o;
    float r;

    cShape(vec4 ox=vec4(0, 0, 0), float rx=1)
    {
        o=ox;
        r=rx;
    }

    float intersect(Ray& ray)
    {
        float dirx=ray.dv.v[0];
        float diry=ray.dv.v[1];
        float dirz=ray.dv.v[2];
        float px=ray.p.v[0];
        float py=ray.p.v[1];
        float pz=ray.p.v[2];
        float ox=o.v[0];
        float oy=o.v[1];
        float oz=o.v[2];

        float a=dirx*dirx+diry*diry+dirz*dirz;
        float b=2*dirx*(px-ox)+2*diry*(py-oy)+2*dirz*(pz-oz);
        float c=ox*ox+oy*oy+oz*oz+px*px+py*py+pz*pz-2*(ox*px+oy*py+oz*pz)-r*r;

        float det=b*b-4*a*c;

        if(det<0.0f)
        {
            return -1.0f;
        }

        float x=((-1.0f*b-sqrtf(det)) / (2.0f*a));

        if(x>epsilon) return x;
        else return 0.0f;
    }

    vec4 surfNorm(vec4& intersection)
    {
        return (intersection-o).normalize();
    }
};

class Field : public Shape
{
public:
    vec4 p;
    vec4 norm;

    Field(vec4 px, vec4 nx)
    {
        p=px;
        norm=nx;
    }

    float intersect(Ray& ray)
    {
        float d=norm.dot(ray.dv);

        if(d==0.0f) return -1.0f;

        float nx=norm.v[0];
        float ny=norm.v[1];
        float nz=norm.v[2];

        float pfx=p.v[0];
        float pfy=p.v[1];
        float pfz=p.v[2];

        float dvx=ray.dv.v[0];
        float dvy=ray.dv.v[1];
        float dvz=ray.dv.v[2];

        float prx=ray.p.v[0];
        float pry=ray.p.v[1];
        float prz=ray.p.v[2];

        double x=-1.0f*(nx*prx-nx*pfx+ ny*pry-ny*pfy+ nz*prz-nz*pfz)/(nx*dvx + ny*dvy + nz*dvz);

        if (x>epsilon) return x;
        if (x>0) return 0.0f;

        return -1;
    }

    vec4 surfNorm(vec4& vec)
    {
        return norm;
    }
};

Scene scene;
static vec4 background[windowWidth * windowHeight];


// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec4 image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -1, -1,   1, -1,  -1, 1,
			                             1, -1,   1,  1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,	0, NULL);     // stride and offset: it is tightly packed

		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;

Sphere *sphere;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	//static vec4 background[windowWidth * windowHeight];
	/*for (unsigned int x = 0; x < windowWidth; x++) {
		for (unsigned int y = 0; y < windowHeight; y++) {
			background[y * windowWidth + x] = vec4((float)x / windowWidth, (float)y / windowHeight, 0, 1);
		}
	}*/

	scene.camera.eye=vec4(0.0f, 0.0f, -300.0f);

	sphere=new Sphere();
	sphere->material=new SmoothMaterial();

	scene.add(sphere);

    scene.render();

    for(int i = 0; i < windowHeight; i++)
    {
        for(int j = 0; j < windowWidth; j++)
        {
            background[(windowHeight-1-i) * windowWidth + j] = vec4(wo[i][j].v[0]/255, wo[i][j].v[1]/255, wo[i][j].v[2]/255, 1.0f);
        }
    }

	fullScreenTexturedQuad.Create( background );

	for(int i = 0; i < windowWidth; i++)
    {
        delete[] wo[i];
    }
    delete[] wo;

	delete sphere->material;
	delete sphere;

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
