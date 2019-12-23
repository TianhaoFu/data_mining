#include<iostream> 
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<typeinfo>
#include<algorithm>
#include<cstring>
#define N 150 
using namespace std;
template<class outT , class inT>
outT convert(const inT &in)
{
	stringstream ss;
	outT out;
	ss << in;
	ss >> out;
	return out;
}
string array[151][5];
double array_double[151][4];

int main()
{
	ifstream inFile("iris_data.csv", ios::in);
	string lineStr;
	vector < vector  < string > > strArray;
	int i , j ;
	i = 0  ; 
	char * end ;
	if ( inFile.fail() )
		cout << "读取文件失败" << endl;
	while ( getline(inFile,lineStr)) 
	{
		j = 0 ;
		stringstream ss(lineStr) ;
		string str;
		vector < string > lineArray ;
		while ( getline(ss,str,','))
		{
			array[i][j] = str;
			j++;
		} 
		i++;
		strArray.push_back(lineArray);
	}
//	检查数据格式 
//	cout<<"bibu"<<strArray[6][4]<<endl;
	for(int i = 0 ; i < 150 ; i  ++ )
	{
		for ( int j = 0 ; j < 4 ; j ++ )
		{
//			cout << "----" <<array[i][j]<<typeid(array[i][j]).name();
			array_double[i][j] = convert<double,string>(array[i][j]);
//			cout << "++++" <<array_double[i][j]<<typeid(array_double[i][j]).name();
		}
		cout << endl;
	}
	
	double min[5],max[5],media[5],q1[5],q3[5];
	double a[N+1];    
	for(int j = 0 ; j <4  ; j ++)
	{
		memset(a,0,sizeof(a));
		for(int i = 0 ; i < 150 ; i ++ )
		{
			a[i+1] =array_double[i][j] ;
		}
		sort(a+1,a+N);
//		for(i=1;i<150;i++)
//			cout<<"Ddaa"<<a[i]<<endl;
		min[j] = a[1];
		max[j] = a[N];
		media[j] = (a[N/2] + a[N/2+1])/2;
//		中位数位置计算公式 
//		长度 ：末-初 + 1
//		中位数 首 + 长度/2 （此题中非整数需额外再+1） 
		int loc1 = (N/2)/2 + 1;
		int loc3 = (N/2+1) + (( N - (N/2+1) + 1) / 2);

		q1[j] = a[loc1] ;
		q3[j] = a[loc3] ;	
	}
	ofstream outFile;
	outFile.open("iris_data_processed.csv",ios::out);
	outFile<<"attribute_name"<<','<<"Min"<<','<<"Q1"<<','<<"Q2"<<','<<"Q3"<<','<<"Max"<<endl;
	outFile << "萼片长度（cm）"<<','<<min[0]<<','<<q1[0]<<','<<media[0]<<','<<q3[0]<<','<<max[0]<<endl; 
	outFile << "萼片宽度（cm）"<<','<<min[1]<<','<<q1[1]<<','<<media[1]<<','<<q3[1]<<','<<max[1]<<endl;
	outFile << "花瓣长度（cm）"<<','<<min[2]<<','<<q1[2]<<','<<media[2]<<','<<q3[2]<<','<<max[2]<<endl;
	outFile << "花瓣宽度（cm）"<<','<<min[3]<<','<<q1[3]<<','<<media[3]<<','<<q3[3]<<','<<max[3]<<endl;
	outFile.close();
	return 0;
}
