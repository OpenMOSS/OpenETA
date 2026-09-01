#pragma once
#include <muda/view/view_base.h>
#include <muda/ext/linear_system/doublet_vector_viewer.h>

namespace muda
{
template <bool IsConst, typename T, int N>
class DoubletVectorViewT : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

    template <bool OtherIsConst, typename U, int M>
    friend class DoubletVectorViewT;

  public:
    using SegmentVector = Eigen::Matrix<T, N, 1>;
    using ConstView     = DoubletVectorViewT<true, T, N>;
    using NonConstView  = DoubletVectorViewT<false, T, N>;
    using ThisView      = DoubletVectorViewT<IsConst, T, N>;

    using ConstViewer    = CDoubletVectorViewer<T, N>;
    using NonConstViewer = DoubletVectorViewer<T, N>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

  protected:
    // vector info
    int m_total_segment_count = 0;

    // doublet info
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

    // subvector info
    int m_subvector_offset = 0;
    int m_subvector_extent = 0;

    // data
    auto_const_t<int>*           m_segment_indices;
    auto_const_t<SegmentVector>* m_segment_values;

  public:
    MUDA_GENERIC DoubletVectorViewT() = default;
    MUDA_GENERIC DoubletVectorViewT(int total_segment_count,
                                    int doublet_index_offset,
                                    int doublet_count,
                                    int total_doublet_count,

                                    int subvector_offset,
                                    int subvector_extent,

                                    auto_const_t<int>* segment_indices,
                                    auto_const_t<SegmentVector>* segment_values)
        : m_total_segment_count(total_segment_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_subvector_offset(subvector_offset)
        , m_subvector_extent(subvector_extent)
        , m_segment_indices(segment_indices)
        , m_segment_values(segment_values)
    {
        MUDA_KERNEL_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorView: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           m_total_doublet_count,
                           doublet_index_offset,
                           doublet_count);

        MUDA_KERNEL_ASSERT(subvector_offset + subvector_extent <= total_segment_count,
                           "DoubletVectorView: out of range, m_total_segment_count=%d, "
                           "your subvector_offset=%d, subvector_extent=%d",
                           m_total_segment_count,
                           subvector_offset,
                           subvector_extent);
    }

    MUDA_GENERIC DoubletVectorViewT(int                total_segment_count,
                                    int                total_doublet_count,
                                    auto_const_t<int>* segment_indices,
                                    auto_const_t<SegmentVector>* segment_values)
        : DoubletVectorViewT(total_segment_count,
                             0,
                             total_doublet_count,
                             total_doublet_count,
                             0,
                             total_segment_count,
                             segment_indices,
                             segment_values)
    {
    }

    template <bool OtherIsConst>
    MUDA_GENERIC DoubletVectorViewT(const DoubletVectorViewT<OtherIsConst, T, N>& other) noexcept
        : m_total_segment_count(other.m_total_segment_count)
        , m_doublet_index_offset(other.m_doublet_index_offset)
        , m_doublet_count(other.m_doublet_count)
        , m_total_doublet_count(other.m_total_doublet_count)
        , m_subvector_offset(other.m_subvector_offset)
        , m_subvector_extent(other.m_subvector_extent)
        , m_segment_indices(other.m_segment_indices)
        , m_segment_values(other.m_segment_values)
    {
    }

    MUDA_GENERIC ConstView as_const() const noexcept
    {
        return ConstView{m_total_segment_count,
                         m_doublet_index_offset,
                         m_doublet_count,
                         m_total_doublet_count,
                         m_subvector_offset,
                         m_subvector_extent,
                         m_segment_indices,
                         m_segment_values};
    }

    MUDA_GENERIC ThisView subview(int offset, int count) const noexcept
    {

        MUDA_KERNEL_ASSERT(offset + count <= m_doublet_count,
                           "DoubletVectorView : offset is out of range, size=%d, your offset=%d",
                           m_doublet_count,
                           offset);

        return ThisView{m_total_segment_count,
                        m_doublet_index_offset + offset,
                        count,
                        m_total_doublet_count,
                        m_subvector_offset,
                        m_subvector_extent,
                        m_segment_indices,
                        m_segment_values};
    }

    MUDA_GENERIC ThisView subview(int offset) const noexcept
    {
        return subview(offset, m_doublet_count - offset);
    }

    MUDA_GENERIC auto subvector(int offset, int extent) const noexcept
    {
        MUDA_KERNEL_ASSERT(offset + extent <= m_subvector_extent,
                           "DoubletVectorView : subvector out of range, extent=%d, your offset=%d, your extent=%d",
                           m_subvector_extent,
                           offset,
                           extent);

        return ThisView{m_total_segment_count,
                        m_doublet_index_offset,
                        m_doublet_count,
                        m_total_doublet_count,
                        m_subvector_offset + offset,
                        extent,
                        m_segment_indices,
                        m_segment_values};
    }

    MUDA_GENERIC ThisViewer viewer() noexcept
    {
        return ThisViewer{m_total_segment_count,
                          m_doublet_index_offset,
                          m_doublet_count,
                          m_total_doublet_count,
                          m_subvector_offset,
                          m_subvector_extent,
                          m_segment_indices,
                          m_segment_values};
    }

    MUDA_GENERIC ConstViewer cviewer() const noexcept
    {
        return ConstViewer{m_total_segment_count,
                           m_doublet_index_offset,
                           m_doublet_count,
                           m_total_doublet_count,
                           m_subvector_offset,
                           m_subvector_extent,
                           m_segment_indices,
                           m_segment_values};
    }

    MUDA_GENERIC int extent() const noexcept { return m_subvector_extent; }

    MUDA_GENERIC int total_extent() const noexcept
    {
        return m_total_segment_count;
    }

    MUDA_GENERIC int subvector_offset() const noexcept
    {
        return m_subvector_offset;
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }
};

template <bool IsConst, typename T>
class DoubletVectorViewT<IsConst, T, 1> : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;

    template <bool OtherIsConst, typename U, int M>
    friend class DoubletVectorViewT;

  protected:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using ConstView    = DoubletVectorViewT<true, T, 1>;
    using NonConstView = DoubletVectorViewT<false, T, 1>;
    using ThisView     = DoubletVectorViewT<IsConst, T, 1>;

    using ConstViewer    = CDoubletVectorViewer<T, 1>;
    using NonConstViewer = DoubletVectorViewer<T, 1>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

  protected:
    int m_total_count = 0;

    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

    int m_subvector_offset = 0;
    int m_subvector_extent = 0;

    auto_const_t<int>* m_indices;
    auto_const_t<T>*   m_values;


  public:
    MUDA_GENERIC DoubletVectorViewT() = default;
    MUDA_GENERIC DoubletVectorViewT(int                total_count,
                                    int                doublet_index_offset,
                                    int                doublet_count,
                                    int                total_doublet_count,
                                    int                subvector_offset,
                                    int                subvector_extent,
                                    auto_const_t<int>* indices,
                                    auto_const_t<T>*   values)
        : m_total_count(total_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_subvector_offset(subvector_offset)
        , m_subvector_extent(subvector_extent)
        , m_indices(indices)
        , m_values(values)
    {
        MUDA_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                    "DoubletVectorView: out of range, m_total_count=%d, "
                    "your doublet_index_offset=%d, doublet_count=%d",
                    m_total_doublet_count,
                    doublet_index_offset,
                    doublet_count);

        MUDA_ASSERT(subvector_offset + subvector_extent <= total_count,
                    "DoubletVectorView: out of range, m_total_count=%d, "
                    "your subvector_offset=%d, subvector_extent=%d",
                    total_count,
                    subvector_offset,
                    subvector_extent);
    }

    MUDA_GENERIC DoubletVectorViewT(int                total_count,
                                    int                total_doublet_count,
                                    auto_const_t<int>* indices,
                                    auto_const_t<T>*   values)
        : DoubletVectorViewT(
            total_count, 0, total_doublet_count, total_doublet_count, 0, total_count, indices, values)
    {
    }

    template <bool OtherIsConst>
    MUDA_GENERIC DoubletVectorViewT(const DoubletVectorViewT<OtherIsConst, T, 1>& other) noexcept
        : m_total_count(other.m_total_count)
        , m_doublet_index_offset(other.m_doublet_index_offset)
        , m_doublet_count(other.m_doublet_count)
        , m_total_doublet_count(other.m_total_doublet_count)
        , m_subvector_offset(other.m_subvector_offset)
        , m_subvector_extent(other.m_subvector_extent)
        , m_indices(other.m_indices)
        , m_values(other.m_values)
    {
    }

    MUDA_GENERIC ConstView as_const() const noexcept
    {
        return ConstView{m_total_count,
                         m_doublet_index_offset,
                         m_doublet_count,
                         m_total_doublet_count,
                         m_subvector_offset,
                         m_subvector_extent,
                         m_indices,
                         m_values};
    }

    MUDA_GENERIC auto subview(int offset, int count) const noexcept
    {
        MUDA_KERNEL_ASSERT(offset + count <= m_doublet_count,
                           "DoubletVectorView : offset is out of range, size=%d, your offset=%d",
                           m_doublet_count,
                           offset);

        return ThisView{m_total_count,
                        m_doublet_index_offset + offset,
                        count,
                        m_total_doublet_count,
                        m_subvector_offset,
                        m_subvector_extent,
                        m_indices,
                        m_values};
    }

    MUDA_GENERIC auto subview(int offset) const noexcept
    {
        return subview(offset, m_doublet_count - offset);
    }

    MUDA_GENERIC auto subvector(int offset, int extent) const noexcept
    {
        MUDA_KERNEL_ASSERT(offset + extent <= m_subvector_extent,
                           "DoubletVectorView : subvector out of range, extent=%d, your offset=%d, your extent=%d",
                           m_subvector_extent,
                           offset,
                           extent);

        return ThisView{m_total_count,
                        m_doublet_index_offset,
                        m_doublet_count,
                        m_total_doublet_count,
                        m_subvector_offset + offset,
                        extent,
                        m_indices,
                        m_values};
    }

    MUDA_GENERIC ThisViewer viewer() noexcept
    {
        return ThisViewer{m_total_count,
                          m_doublet_index_offset,
                          m_doublet_count,
                          m_total_doublet_count,

                          m_subvector_offset,
                          m_subvector_extent,

                          m_indices,
                          m_values};
    }

    MUDA_GENERIC ConstViewer cviewer() const noexcept
    {
        return ConstViewer{m_total_count,
                           m_doublet_index_offset,
                           m_doublet_count,
                           m_total_doublet_count,
                           m_subvector_offset,
                           m_subvector_extent,
                           m_indices,
                           m_values};
    }


    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC int extent() const noexcept { return m_subvector_extent; }
    MUDA_GENERIC int total_extent() const noexcept { return m_total_count; }

    MUDA_GENERIC int subvector_offset() const noexcept
    {
        return m_subvector_offset;
    }
};

template <typename T, int N>
using DoubletVectorView = DoubletVectorViewT<false, T, N>;
template <typename T, int N>
using CDoubletVectorView = DoubletVectorViewT<true, T, N>;
}  // namespace muda

namespace muda
{
template <typename Ty, int N>
struct read_only_view<DoubletVectorView<Ty, N>>
{
    using type = CDoubletVectorView<Ty, N>;
};

template <typename Ty, int N>
struct read_write_view<CDoubletVectorView<Ty, N>>
{
    using type = DoubletVectorView<Ty, N>;
};
}  // namespace muda

#include "details/doublet_vector_view.inl"
