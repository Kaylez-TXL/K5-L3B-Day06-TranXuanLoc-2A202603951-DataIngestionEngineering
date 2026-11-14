# Báo cáo cá nhân — Day 6

**Họ tên:** Trần Xuân Lộc · **MSSV:** 2A202603951 · **Lớp:** L3B

## 1. Dữ liệu và cách xử lý

Em sử dụng đồ án tốt nghiệp do em hoàn thành: **“Nghiên cứu quang phổ của hạt nano lõi–vỏ Au@Ag và ứng dụng học máy trong dự đoán đặc tính quang học”**. Mục tiêu bài thực hành là tìm các đoạn bằng chứng liên quan đến câu hỏi và so sánh ba cách chuẩn bị dữ liệu.

PDF có 70 trang. Em dùng toàn bộ 54 trang nội dung chính (trang 12–65), giữ cả trang có hình và bảng; không tính phần đầu và danh mục tham khảo. Nguồn, quyền sử dụng, danh sách trang và mã kiểm tra file được lưu trong [manifest](../../data/manifest.json). Các nguồn tham khảo trong đồ án được giữ nguyên trích dẫn.

Pipeline gồm: đọc PDF → chuẩn hóa văn bản → chia đoạn (chunk) → tạo embedding → lưu pgvector → tìm ba đoạn liên quan nhất. Sau chuẩn hóa có 98.335 ký tự. Chương trình chuẩn hóa Unicode và khoảng trắng, giữ số trang để truy nguồn. Các trang ít chữ được ghi cảnh báo, không tự loại bỏ. Phần trích xuất hiện chỉ xử lý văn bản, chưa nhận diện nội dung đồ thị và còn một số lỗi tách từ tiếng Việt.

Mỗi chunk có `document_id`, `source`, `page` và `chapter` để truy nguồn, lọc chương. Giá trị chương lấy từ cấu trúc tài liệu, không lấy từ đáp án. Chương trình cũng lưu vị trí ký tự để đối chiếu đoạn gốc.

Em dùng **intfloat/multilingual-e5-small** chạy trên CPU, vector 384 chiều, giới hạn 512 token. Câu hỏi có tiền tố `query: `, đoạn tài liệu có `passage: `; vector được chuẩn hóa L2. Đầu vào quá dài sẽ báo lỗi thay vì tự cắt. Cả ba cấu hình dùng cùng model và revision.

Dữ liệu lưu trong PostgreSQL 16.16 + pgvector 0.8.7, bảng `day6_chunks`. Tìm kiếm dùng cosine chính xác, không dùng chỉ mục gần đúng. Khi nạp lại, chương trình thay toàn bộ dữ liệu của từng cấu hình trong một transaction, xóa cả chunk cũ không còn dùng. Đã kiểm tra nạp lại không tăng bản ghi trùng, dữ liệu đọc được từ kết nối khác và lỗi ghi không làm mất dữ liệu cũ.

Môi trường chạy: Python 3.14.6, sentence-transformers 6.3.0, transformers 5.23.0, torch 2.15.0, pypdf 6.21.4, psycopg 3.3.8 và python-dotenv 1.2.4. Model revision và thông tin lần chạy được lưu trong [metrics.json](../../results/metrics.json). Phí API là 0; chưa tính điện, thiết bị và thời gian làm bài.

## 2. Cấu hình và bộ câu hỏi

| Cấu hình | Cách chia chunk | Lọc khi tìm kiếm |
|---|---|---|
| A | Tối đa 500 ký tự, overlap 50 ký tự | Không |
| B | Gom theo dòng trống, tiêu đề, bullet và kết thúc câu; tối đa 500 ký tự, không overlap | Không |
| C | Dùng đúng chunk và vector của B | Lọc chương khi câu hỏi yêu cầu |

A–B so sánh cách chia đoạn; B–C chỉ thay việc lọc. Corpus, model, vector store, metric và bộ câu hỏi được giữ nguyên. B suy ra cấu trúc từ văn bản PDF nên không bảo đảm giữ đúng đoạn văn gốc. Các chunk không nối qua trang.

[Bộ câu hỏi](../../benchmark/questions.jsonl) có 20 câu, chia trước thành 15 dev và 5 test. Có 4 câu yêu cầu lọc chương, gồm q19 thuộc test. Em đã kiểm tra lại câu hỏi, đáp án và bằng chứng theo tài liệu. Việc chia trong cùng một đồ án vẫn có nguy cơ giao thoa chủ đề.

Em chọn cấu hình theo Evidence Hit@3 trên dev, sau đó Hit@1, rồi thời gian ingest nếu bằng điểm. Cấu hình **A được chọn trước khi chạy test**, ghi trong [selection.json](../../results/selection.json). Em không chỉnh cấu hình theo kết quả test.

## 3. Kết quả

Hit@1 xét đoạn đứng đầu; Evidence Hit@3 xét ba đoạn đầu. Cách chấm tự động yêu cầu một chunk đúng nguồn/trang và chứa đầy đủ trích dẫn đã ghi. Cách chấm này có thể bỏ sót đoạn đủ ý nhưng diễn đạt khác hoặc bằng chứng bị tách qua nhiều chunk.

| Cấu hình | Tập | Hit@1 | Evidence Hit@3 |
|---|---|---|---|
| A | Dev | 11/15 | 14/15 |
| A | Test | 3/5 | 3/5 |
| B | Dev | 12/15 | 13/15 |
| B | Test | 1/5 | 1/5 |
| C | Dev | 12/15 | 13/15 |
| C | Test | 1/5 | 1/5 |

| Cấu hình | Số chunk | Độ dài min / trung bình / max (ký tự) | Tổng token document | Ingest (giây) |
|---|---:|---|---:|---:|
| A | 237 | 58 / 453,5 / 500 | 32.631 | 7,912 |
| B | 268 | 4 / 366,9 / 500 | 29.888 | 7,205 |
| C | 268 | 4 / 366,9 / 500 | 29.888 | 0,150 |

Lần chạy ngày 14/11/2026 lúc 11:26 (UTC+7) đo được tổng thời gian pipeline 27,41 giây, không gồm phần import trước bộ đếm và kiểm tra tích hợp riêng. C dùng lại embedding của B nên thời gian C chỉ tính phần ghi dữ liệu, không thể so trực tiếp như chi phí chạy độc lập. Sau nạp lại, số bản ghi vẫn là 237/268/268.

### Đánh giá thủ công

Em đã chấm 90 mục trong [bảng đánh giá thủ công](../../results/manual_audit.csv): 30 coherence, 15 grounding và 45 metadata. AI hỗ trợ đề xuất nhãn ban đầu; em kiểm tra, điều chỉnh và xác nhận kết quả chấm. Coherence và grounding chấm 0–2; metadata chấm 0/1. Mẫu coherence dùng cùng 10 vị trí nguồn đã chọn trước, còn grounding và metadata dùng 5 câu test cho từng cấu hình.

| Cấu hình | Coherence trung bình /2 (10 mẫu) | Metadata đủ và đúng | Top-3 tuân thủ filter (test) | Grounding: số câu mức 0/1/2; trung bình /2 |
|---|---:|---|---|---|
| A | 0,9 | 15/15 | 1/1 | 0/1/4; 1,8 |
| B | 0,9 | 15/15 | 1/1 | 0/2/3; 1,6 |
| C | 0,9 | 15/15 | 1/1 | 0/2/3; 1,6 |

A và B không áp filter nhưng kết quả câu test có điều kiện vẫn thuộc đúng chương. C có áp filter; với một câu test có filter, chưa thể kết luận bộ lọc giúp tăng điểm. `page` hữu ích để kiểm tra nguồn, `chapter` giúp giới hạn phạm vi nhưng không giải quyết lỗi chia đoạn.

Grounding đánh giá độ đủ của bằng chứng trong cả top-3, không phải độ chính xác câu trả lời LLM. Vì vậy, grounding có thể đạt 2 khi bằng chứng nằm ở hai chunk, dù Evidence Hit@3 vẫn bằng 0. Người chấm là em, cũng là người làm bài, nên kết quả có thể chịu ảnh hưởng chủ quan.

## 4. Phân tích và kết luận

Em chọn **A** vì có Evidence Hit@3 dev cao nhất (14/15). B có Hit@1 dev cao hơn (12/15 so với 11/15) nhưng Hit@3 thấp hơn. C không tăng điểm so với B trong lần thử này. Mẫu coherence cũng chưa cho thấy cách B giữ ý tốt hơn A. A dùng ít chunk hơn nhưng tổng token cao hơn do overlap.

Hai trường hợp lỗi đáng chú ý:

- **q17, B/C, trang 51:** phần nói khe hở 2 nm và giá trị FoM 24 bị chia qua hai chunk. Top-3 đủ để hiểu đáp án, nhưng không chunk nào chứa trọn bằng chứng nên Hit@3 tự động bằng 0.
- **q20, B/C, trang 63:** tìm được ý dùng 1D-CNN làm mô hình nền nhưng thiếu phần lấy dữ liệu thực nghiệm để tinh chỉnh trọng số. Vì vậy grounding chỉ đạt 1 và Hit@3 bằng 0.

Bài còn hạn chế: chỉ có 5 câu test, cùng một tài liệu, lỗi trích chữ PDF, chưa đọc được đồ thị và thời gian chỉ đo một lần. Bộ câu hỏi và lý do chấm có AI hỗ trợ; em đã kiểm tra lại nhưng vẫn có nguy cơ thiên lệch. Điểm khớp trích dẫn chưa bao quát hết bằng chứng tương đương về ý nghĩa.

Hướng cải thiện là xác định ranh giới đoạn văn chính xác hơn, thử chia theo token/câu trên dev và bổ sung tập test độc lập. Kết quả hiện tại chỉ áp dụng cho corpus và lần thực nghiệm này.

Em sử dụng Codex để hỗ trợ viết mã, soạn câu hỏi và phân tích kết quả. Em đã kiểm tra 20 câu hỏi, chấm lại 90 mục đánh giá và chịu trách nhiệm về nội dung bài nộp.
