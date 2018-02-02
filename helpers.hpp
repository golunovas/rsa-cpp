#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <chrono>

class Timer {
private:
	std::chrono::time_point<std::chrono::system_clock> t1;
public:
	inline void start() {
		t1 = std::chrono::system_clock::now();
	} 
	inline double stop() {
		auto t2 = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
};

#endif // HELPERS_HPP_