#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace strata {

// Remembers which nodes one search has already looked at.
// Instead of clearing the whole array for each search, every search gets a new
// number, and a node counts as visited only if its slot holds the current number.
class VisitedList {
public:
    explicit VisitedList(std::size_t capacity) : marks_(capacity, 0) {}

    void start_new_search() {
        ++current_;
        if (current_ == 0) {  // the counter wrapped around, so clear once
            std::fill(marks_.begin(), marks_.end(), 0);
            current_ = 1;
        }
    }

    // Returns true if the node was already visited; otherwise marks it.
    bool check_and_mark(std::uint32_t node) {
        if (marks_[node] == current_) {
            return true;
        }
        marks_[node] = current_;
        return false;
    }

private:
    std::vector<std::uint16_t> marks_;
    std::uint16_t current_ = 0;
};

// Keeps spare VisitedLists so searches don't allocate a new one every time.
class VisitedPool {
public:
    explicit VisitedPool(std::size_t capacity) : capacity_(capacity) {}

    std::unique_ptr<VisitedList> take() {
        {
            std::lock_guard lock(mutex_);
            if (!spare_.empty()) {
                auto list = std::move(spare_.back());
                spare_.pop_back();
                return list;
            }
        }
        return std::make_unique<VisitedList>(capacity_);
    }

    void give_back(std::unique_ptr<VisitedList> list) {
        std::lock_guard lock(mutex_);
        spare_.push_back(std::move(list));
    }

private:
    std::size_t capacity_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedList>> spare_;
};

// Takes a list from the pool and returns it automatically when done.
class VisitedHandle {
public:
    explicit VisitedHandle(VisitedPool& pool) : pool_(pool), list_(pool.take()) {}
    ~VisitedHandle() { pool_.give_back(std::move(list_)); }
    VisitedHandle(const VisitedHandle&) = delete;
    VisitedHandle& operator=(const VisitedHandle&) = delete;

    VisitedList& operator*() { return *list_; }
    VisitedList* operator->() { return list_.get(); }

private:
    VisitedPool& pool_;
    std::unique_ptr<VisitedList> list_;
};

}  // namespace strata
