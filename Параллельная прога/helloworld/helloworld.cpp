#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include <string.h>
#include <string>
#include <set>
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace std;

bool isWord(const string& str) {

    if (str.find('.') != string::npos && str.find("text.txt") != string::npos) {
        return false;
    }

    static const set<string> articles = {"a", "an", "the", "mpi"};

    string lower_str = str;
    transform(lower_str.begin(), lower_str.end(), lower_str.begin(), [](unsigned char c){ return tolower(c); });

    return articles.find(lower_str) == articles.end();
}

string clean_word(const string& word) {
    string cleaned;
    for (char ch : word) {
        if (isalnum(ch)) {
            cleaned += (char)tolower(ch);
        }
    }
    return cleaned;
}

// Функция для разделения строки на уникальные слова
void split_into_words(const string& str, set<string>& unique_words) {
    istringstream iss(str);
    string word;
    while (iss >> word) {
        if (isWord(word)) {
            unique_words.insert(clean_word(word));

            //cout << clean_word(word) << " " << unique_words.size() << endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time;

    string text;
    string filename = "text.txt";

    if (rank == 0) {
        double start_time2 = MPI_Wtime();
        ifstream inputFile(filename);

        string line;
        getline(inputFile, line);

        set<string> uniqueWords;

        istringstream stream(line);
        string word;

        while (stream >> word) {
            if (isWord(word)) {
                uniqueWords.insert(clean_word(word));
            }
        }

        cout << "Total unique (seq): " << uniqueWords.size() << endl;

        inputFile.close();

        cout << "seq: " << MPI_Wtime() - start_time2 << endl;
    }

    if (rank == 0) {
        start_time = MPI_Wtime();

        ifstream file(filename);
        stringstream buffer;
        buffer << file.rdbuf();
        text = buffer.str();
    }

    int text_length = text.size();
    MPI_Bcast(&text_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char* text_buffer = new char[text_length + 1];
    if (rank == 0) {
        strcpy(text_buffer, text.c_str());
    }
    
    MPI_Bcast(text_buffer, text_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Разбиение текста на части для каждого процесса
    int local_length = text_length / size;
    int start = rank * local_length;
    int end = (rank == size - 1) ? text_length : (rank + 1) * local_length;

    // Обработка границ, чтобы не разрывать слова
    if (rank > 0) {
        while (start > 0 && text_buffer[start] != ' ') {
            start--;
        }
    }

    if (rank < size - 1) {
        while (end < text_length && text_buffer[end] != ' ') {
            end++;
        }
    }

    // Получение локального текста
    string local_text(text_buffer + start, end - start);
    delete[] text_buffer;

    // Подсчет уникальных слов в локальном тексте
    set<string> local_unique_words;
    split_into_words(local_text, local_unique_words);

    // Сбор уникальных слов на главном процессе
    set<string> global_unique_words;

    int local_count = local_unique_words.size();
    int* counts = nullptr;

    if (rank == 0) {
        counts = new int[size];
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Сбор всех уникальных слов
    vector<string> all_words;
    if (rank == 0) {
        all_words.reserve(text_length);
    }

    for (const auto& word : local_unique_words) {
        int word_length = word.length();
        char* word_buffer = new char[word_length + 1];
        strcpy(word_buffer, word.c_str());
        
        if (rank == 0) {
            all_words.push_back(word);
        } else {
            MPI_Send(word_buffer, word_length + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        delete[] word_buffer;
    }

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < counts[i]; j++) {
                char word_buffer[100];
                MPI_Recv(word_buffer, sizeof(word_buffer), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                all_words.push_back(word_buffer);
            }
        }
        
        for (const auto& word : all_words) {
            global_unique_words.insert(word);
        }

        cout << "Total unique (paral): " << global_unique_words.size() << endl;
        delete[] counts;

        cout << "paral: " << MPI_Wtime() - start_time << endl;
    }
    
    MPI_Finalize();
    return 0;
}

















// #include <iostream>
// #include <mpi.h>
// #include <cstdlib>
// #include <ctime>
// #include <vector>
// using namespace std;

// #define N 2

// void initmx(vector<int>& matrix, int shift) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             matrix[i * N + j] = i + j + shift;
//         }
//     }
// }

// void seqmul(const vector<int>& A, const vector<int>& B, vector<int>& result) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             result[i * N + j] = 0;
//             for (int k = 0; k < N; k++) {
//                 result[i * N + j] += A[i * N + k] * B[k * N + j];
//             }
//         }
//     }
// }

// void paralmul(int rank, int size, const vector<int>& A, const vector<int>& B, vector<int>& local_result, int rows_for_process) {
//     for (int i = 0; i < rows_for_process; i++) {
//         for (int j = 0; j < N; j++) {
//             local_result[i * N + j] = 0;
//             for (int k = 0; k < N; k++) {
//                 local_result[i * N + j] += A[i * N + k] * B[k * N + j];
//             }
//         }
//     }
// }

// void print_matrix(const vector<int>& matrix) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             cout << matrix[i * N + j] << " ";
//         }
//         cout << "\n";
//     }
// }

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     double start_time;
//     vector<int> A(N * N);
//     vector<int> B(N * N);
//     vector<int> result(N * N);
//     vector<int> result_parallel(N * N);
    
//     if (rank == 0) {
//         initmx(A, 0);
//         initmx(B, 1);

//         print_matrix(A);
//         cout << "\n";
//         print_matrix(B);
//         cout << "\n";
//     }

//     if (rank == 0) {
//         start_time = MPI_Wtime();
//         seqmul(A, B, result);
//         double end_time = MPI_Wtime();

//         //cout << "\nSeq: " << end_time - start_time << " s.\n";
//         //print_matrix(result);
//         print_matrix(result);
//         cout << "\n";
//     }

//     MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    
//     int rows_per_process = N / size;
//     int extra_rows = N % size;
//     int rows_for_process = rows_per_process + (rank < extra_rows ? 1 : 0);
    
//     vector<int> A_sub(rows_for_process * N);
//     vector<int> C_local(rows_for_process * N);

//     int sendcounts[size];
//     int displacements[size];

//     for (int i = 0; i < size; i++) {
//         sendcounts[i] = rows_per_process * N;
//         if (i < extra_rows) {
//             sendcounts[i] += N;
//         }
//         displacements[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * N;
//     }

//     start_time = MPI_Wtime();
//     MPI_Scatterv(A.data(), sendcounts, displacements, MPI_INT, A_sub.data(), rows_for_process * N, MPI_INT, 0, MPI_COMM_WORLD);

//     paralmul(rank, size, A_sub, B, C_local, rows_for_process);

//     MPI_Gatherv(C_local.data(), rows_for_process * N, MPI_INT, result_parallel.data(), sendcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
//     double end_time = MPI_Wtime();

//     if (rank == 0) {
//         //cout << "\nParal: " << end_time - start_time << " s.\n";
//         print_matrix(result_parallel);
//     }

//     MPI_Finalize();
//     return 0;
// }



// #include <iostream>
// #include <mpi.h>
// #include <vector>
// #include <cstdlib>
// #include <ctime>
// #include <algorithm>

// using namespace std;
// int RDVALUE_ = 1;

// int RD(void) {
//     int A;
//     RDVALUE_ = (RDVALUE_ + (int)(unsigned long long)(&A)) * 1103515245 + 12345;
//     return RDVALUE_ / 31;
// }

// int main(int argc, char** argv) {

//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const long N = 100000000;
//     long chunk_size = N / size;
//     vector<int> local_array(chunk_size);
//     vector<int> array(N);

//     if (rank == 0){
//         srand(time(nullptr));
//         for (long i = 0; i < N; ++i) {
//             int A;
//             array[i] = RD();
//             //cout << array[i] << " ";
//         } cout << endl;
//     }

//     if (rank == 0){
//         double start_time0, end_time0;
//         start_time0 = MPI_Wtime();

//         int max_value = *max_element(array.begin(), array.end());
//         cout << "Seq maximum: " << max_value << endl;
//         end_time0 = MPI_Wtime();
//         cout << "Time: " << end_time0 - start_time0 << endl;
//     }

//     double start_time, end_time;
//     start_time = MPI_Wtime();

//     MPI_Scatter(array.data(), chunk_size, MPI_INT, local_array.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
//     int local_max = *max_element(local_array.begin(), local_array.end());
    
//     int global_max;
//     MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

//     end_time = MPI_Wtime();

//     if (rank == 0) {
//         cout << "Paral maximum: " << global_max << endl;
//         cout << "Time: " << end_time - start_time << endl;
//     }

//     MPI_Finalize();
//     return 0;
// }






// #define POINTCOUNT 1000000000

// int main(int argc, char **argv){
//     MPI_Init(&argc, &argv);
    
//     int rank, size;

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int points_per_process = (double)POINTCOUNT / size;
//     srand(time(NULL) + rank);
//     double start_time = MPI_Wtime();

//     int localcount = 0;

//     for (int i=0; i<points_per_process; i++){
//         double x = (double)rand() / RAND_MAX;
//         double y = (double)rand() / RAND_MAX;
//         if (x * x + y * y <= 1.0) {
//             localcount++;
//         }
//     }

//     int globalcount;
//     MPI_Reduce(&localcount, &globalcount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         double pi = (double)globalcount / POINTCOUNT * 4;
//         double end_time = MPI_Wtime();

//         cout << endl;
//         cout << "Approximation of Pi: " << pi << endl;
//         cout << "Time taken: " << (end_time - start_time) << " seconds." << endl;
//     }

//     MPI_Finalize();
// }




// #include <iostream>
// #include <mpi.h>
// #include <iomanip>
// using namespace std;

// int main(int argc, char **argv) {
//     cout << fixed << setprecision(6);
//     cout << endl;

//     int rank, size;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int SIZE = size;
//     MPI_Request request;

//     int data[4 * SIZE];
//     int segment[SIZE];
//     int gathered[4 * SIZE];

//     if (rank == 0) {
//         for (int i = 0; i < 4 * SIZE; i++) {
//             data[i] = i + 1;
//         }
//         cout << "Process 0 scatter data: ";
//         for (int i = 0; i < 4 * SIZE; i++) {
//             cout << data[i] << " ";
//         }
//         cout << endl;
//     }


//     // Синхронщина
//     double start_time = MPI_Wtime();
//     MPI_Scatter(data, 4, MPI_INT, segment, 4, MPI_INT, 0, MPI_COMM_WORLD);

//     for (int i = 0; i < 4; i++) {
//         segment[i] *= rank;
//     }

//     MPI_Gather(segment, 4, MPI_INT, gathered, 4, MPI_INT, 0, MPI_COMM_WORLD);
//     double sync_time = MPI_Wtime() - start_time;

//     if (rank == 0) {
//         cout << "Sync: " << sync_time << " sec." << endl;
//         cout << "Process 0 gathered data: ";
//         for (int i = 0; i < 4 * SIZE; i++) {
//             cout << gathered[i] << " ";
//         }
//         cout << endl;
//     }


//     // Асинхронщина
//     MPI_Barrier(MPI_COMM_WORLD);

//     start_time = MPI_Wtime();
//     MPI_Iscatter(data, 4, MPI_INT, segment, 4, MPI_INT, 0, MPI_COMM_WORLD, &request);

//     for (int i = 0; i < 4; i++) {
//         segment[i] *= rank;
//     }

//     MPI_Wait(&request, MPI_STATUS_IGNORE);

//     MPI_Igather(segment, 4, MPI_INT, gathered, 4, MPI_INT, 0, MPI_COMM_WORLD, &request);

//     double async_time = MPI_Wtime() - start_time;

//     if (rank == 0) {
//         cout << "Async: " << async_time << " sec." << endl;
//         cout << "Process 0 gathered data: ";
//         for (int i = 0; i < 4 * SIZE; i++) {
//             cout << gathered[i] << " ";
//         }
//         cout << endl;
//     }

//     MPI_Finalize();
// }



// #include <mpi.h>
// #include <iostream>
// #include <string>
// using namespace std;

// #define SIZE 10

// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     int arr[SIZE];
//     int local_sum = 0, global_sum = 0;
//     int local_max = 0, global_max = 0;

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         cout << "Process 0 broadcasting data...\n";
//         for (int i = 0; i < SIZE; i++) {
//             arr[i] = i + 1;
//             //cout << arr[i] << " ";
//         }
//     }

//     MPI_Bcast(arr, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

//     if (rank != 0){
//         for (int i = 0; i < SIZE; i++) {
//             arr[i] *= rank;

//             //cout << arr[i] << " ";
//         }

//         for(int i = 0; i < SIZE; i++) {
//             local_sum += arr[i];

//             if (arr[i] > local_max) {
//                 local_max = arr[i];
//             }
//         }
//     }

//     MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//     MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

//     MPI_Barrier(MPI_COMM_WORLD);

//     if(rank == 0) {
//         cout << endl << "max: " << global_max << endl;
//         cout << "sum: " << global_sum << endl << endl;
//     }

//     MPI_Finalize();
// }


// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     string msg;
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int MSG_LEN = 20;
//     char message[MSG_LEN];

//     if (rank == 0) {
//         for (int i = 1; i < size; ++i) {
//             msg = "Hello from process 0 to ";
//             msg += to_string(i);
//             MPI_Send(msg.c_str(), MSG_LEN, MPI_CHAR, i, 0, MPI_COMM_WORLD);
//         }

//     } else {
//         MPI_Recv(message, MSG_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         cout << "Process " << rank << " received message: " << message << endl;
//     }

//     MPI_Finalize();
//     return 0;
// }

// #include <iostream>
// #include <mpi.h>


//int main(int argc, char** argv) {

    // // Инициализация MPI
    // MPI_Init(&argc, &argv);

    // // Получаем количество процессов
    // int treads_num;
    // MPI_Comm_size(MPI_COMM_WORLD, &treads_num);

    // // Получаем ранг (идентификатор) текущего процесса
    // int treads_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &treads_rank);

    // // Каждый процесс выводит свое сообщение
    // cout << "Hello from process " << treads_rank << " out of " << treads_num << " processes" << endl;

    // // Завершение работы MPI
    // MPI_Finalize();

    // return 0;
//}

// mpiexec -n 4 helloworld.exe