#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include<algorithm>

//分割字符串
std::vector<std::string> split(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern;//扩展字符串以方便操作

    for(int i=0; i<str.size(); i++)
    {
        pos=str.find(pattern,i);
        if(pos<str.size())
        {
            std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
    return result;
}

//删除重复元素
void elimDups(std::vector<std::string> &word){
    // sort要包含头文件algorithm
    std::sort(word.begin(),word.end());
    //unique 重排输入范围,使得每个单词只出现一次
    //返回排列的不重复区域后的一个位置的迭代器
    std::vector<std::string>::iterator end_unique = std::unique(word.begin(),word.end());
    //删除重复单词
    word.erase(end_unique,word.end());
}

//定义一个类用来存储标字段及对应记录不同种类数目
class Location{
private:
    std::string loc;
    int num;
public:
    Location(std::string loc,int num){
        this->loc = loc;
        this->num = num;
    }
    Location(Location &l){loc=l.loc; num=l.num;}
    int get_num(){return num;}
    std::string get_loc(){return loc;}
};



int main()
{
    std::string s;
    bool flag = true;
    std::ifstream fin;
    std::ofstream fout;
    //field字段数组,对应 区,市,省,国家等
    //tuple元组
    //l1-4 四个字段分别对应的记录数
    std::vector<std::string> field,tuple,l1,l2,l3,l4,l[4];

    fin.open("concept_input.txt");
    fout.open("concept_output.txt");

    //==读入数据
    while(!fin.eof())
    {
        std::getline(fin,s); 
        if(s[0]=='#'||s[0]==NULL)
            continue;
        else{
            if(flag){// 分割第一行
                field=split(s," ");     
                flag = false;
            }
            else{   //将之后的每行的元素添加到content中
                tuple=split(s," ");
                l[0].push_back(tuple[0]);
                l[1].push_back(tuple[1]);
                l[2].push_back(tuple[2]);
                l[3].push_back(tuple[3]);           
            }
        }   
    }

    //==消除重复 之后l1的大小即为其元素个数
    elimDups(l1);
    elimDups(l2);
    elimDups(l3);
    elimDups(l4);


    //将字段field与其对应的值一一对应起来
    Location A[5]={Location(field[0],l[0].size()),//哨兵
        Location(field[0],l[0].size()),
        Location(field[1],l[1].size()),
        Location(field[2],l[2].size()),
        Location(field[3],l[3].size())};


    //直接插入排序
    int i,j;
    for(i=2; i<=4; i++){
        A[0] = A[i];
        for(j = i -1; A[0].get_num()<A[j].get_num(); --j )
            A[j+1] = A[j];
        A[j+1] = A[0];      
    }


    //输出
    std::cout<<"自动产生这些属性的概念分层从低到高为:"<<std::endl;
    for(i=1; i<=4; ++i)
    {
        std::cout<<A[i].get_loc()<<" < ";
        //输出到文件
        fout<<A[i].get_loc()<<" < ";
    }

    fin.close();
    fout.close();
    std::cout<<std::endl;
    system("pause");
    return 0;
}
