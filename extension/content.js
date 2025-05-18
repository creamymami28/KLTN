const API_URL = "http://localhost:8080/predict";

async function checkSpam(commentText) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comment: commentText }),
  });
  const data = await res.json();
  return data.prediction === "Spam";
}

function processCommentElement(commentEl) {
  // Lấy phần chứa text comment
  const textEl = commentEl.querySelector(".update-components-text");
  if (!textEl || textEl.getAttribute("data-spam-checked") === "true") return;

  const commentText = textEl.innerText.trim();
  if (!commentText) return;

  textEl.setAttribute("data-spam-checked", "true");
  console.log("🔍 Phát hiện comment:", commentText);

  checkSpam(commentText).then((isSpam) => {
    if (isSpam) {
        console.log("🚨 Đánh dấu spam:", commentText);

        const spamTag = document.createElement("span");
        spamTag.innerText = " spam";
        spamTag.style.color = "red";
        spamTag.style.fontWeight = "bold";
        spamTag.style.marginLeft = "6px";
        textEl.appendChild(spamTag);
        textEl.style.color = "red";
        textEl.style.fontWeight = "bold";
    }
  });
}

// MutationObserver theo dõi DOM thay đổi
const observer = new MutationObserver(() => {
  console.log("👀 DOM thay đổi, quét comment...");
  // Chọn tất cả các phần tử comment theo class chính xác
  const comments = document.querySelectorAll(".comments-comment-item__main-content");
  comments.forEach(processCommentElement);
});

// Bắt đầu theo dõi
observer.observe(document.body, { childList: true, subtree: true });

// Quét lần đầu khi load trang xong
window.addEventListener("load", () => {
  console.log("🚀 Trang đã tải, quét comment...");
  const comments = document.querySelectorAll(".comments-comment-item__main-content");
  comments.forEach(processCommentElement);
});
