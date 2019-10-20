#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include<algorithm>

//�ָ��ַ���
std::vector<std::string> split(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern;//��չ�ַ����Է������

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

//ɾ���ظ�Ԫ��
void elimDups(std::vector<std::string> &word){
    // sortҪ����ͷ�ļ�algorithm
    std::sort(word.begin(),word.end());
    //unique �������뷶Χ,ʹ��ÿ������ֻ����һ��
    //�������еĲ��ظ�������һ��λ�õĵ�����
    std::vector<std::string>::iterator end_unique = std::unique(word.begin(),word.end());
    //ɾ���ظ�����
    word.erase(end_unique,word.end());
}

//����һ���������洢���ֶμ���Ӧ��¼��ͬ������Ŀ
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
    //field�ֶ�����,��Ӧ ��,��,ʡ,���ҵ�
    //tupleԪ��
    //l1-4 �ĸ��ֶηֱ��Ӧ�ļ�¼��
    std::vector<std::string> field,tuple,l1,l2,l3,l4,l[4];

    fin.open("concept_input.txt");
    fout.open("concept_output.txt");

    //==��������
    while(!fin.eof())
    {
        std::getline(fin,s); 
        if(s[0]=='#'||s[0]==NULL)
            continue;
        else{
            if(flag){// �ָ��һ��
                field=split(s," ");     
                flag = false;
            }
            else{   //��֮���ÿ�е�Ԫ����ӵ�content��
                tuple=split(s," ");
                l[0].push_back(tuple[0]);
                l[1].push_back(tuple[1]);
                l[2].push_back(tuple[2]);
                l[3].push_back(tuple[3]);           
            }
        }   
    }

    //==�����ظ� ֮��l1�Ĵ�С��Ϊ��Ԫ�ظ���
    elimDups(l1);
    elimDups(l2);
    elimDups(l3);
    elimDups(l4);


    //���ֶ�field�����Ӧ��ֵһһ��Ӧ����
    Location A[5]={Location(field[0],l[0].size()),//�ڱ�
        Location(field[0],l[0].size()),
        Location(field[1],l[1].size()),
        Location(field[2],l[2].size()),
        Location(field[3],l[3].size())};


    //ֱ�Ӳ�������
    int i,j;
    for(i=2; i<=4; i++){
        A[0] = A[i];
        for(j = i -1; A[0].get_num()<A[j].get_num(); --j )
            A[j+1] = A[j];
        A[j+1] = A[0];      
    }


    //���
    std::cout<<"�Զ�������Щ���Եĸ���ֲ�ӵ͵���Ϊ:"<<std::endl;
    for(i=1; i<=4; ++i)
    {
        std::cout<<A[i].get_loc()<<" < ";
        //������ļ�
        fout<<A[i].get_loc()<<" < ";
    }

    fin.close();
    fout.close();
    std::cout<<std::endl;
    system("pause");
    return 0;
}
