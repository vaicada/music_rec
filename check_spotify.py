"""
Kiểm tra Spotify Client ID + Secret bằng Python thuần.
Không cần cài thêm thư viện nào ngoài 'requests'.

Chạy: py check_spotify.py
"""

import base64
import requests

# ======================================================
# 1. THAY ĐỔI 2 DÒNG NÀY ĐỂ KIỂM TRA BẤT KỲ CREDENTIALS NÀO
# ======================================================
CLIENT_ID     = "0054a24f2fc643c69d56d020dd5f70be"
CLIENT_SECRET = "98b4a4b772ad4eca934a92ca60c246a0"
# ======================================================


def check_spotify(client_id: str, client_secret: str):
    print("=" * 60)
    print(f"  Kiểm tra Spotify Credentials")
    print(f"  Client ID: {client_id[:8]}...{client_id[-4:]}")
    print("=" * 60)

    # ── Bước 1: Lấy Access Token ──────────────────────────────────
    # Spotify dùng HTTP Basic Auth: base64("client_id:client_secret")
    credentials = f"{client_id}:{client_secret}"
    encoded     = base64.b64encode(credentials.encode()).decode()

    print("\n[1] Đang xác thực với Spotify...")
    try:
        auth_response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {encoded}"},
            data={"grant_type": "client_credentials"},
            timeout=10
        )
    except requests.exceptions.ConnectionError:
        print("    ❌ Không kết nối được Internet!")
        return
    except requests.exceptions.Timeout:
        print("    ❌ Timeout — Spotify không phản hồi!")
        return

    if auth_response.status_code != 200:
        print(f"    ❌ Xác thực THẤT BẠI")
        print(f"    HTTP {auth_response.status_code}: {auth_response.text}")
        if auth_response.status_code == 400:
            print("    → Client ID hoặc Client Secret SAI")
        elif auth_response.status_code == 401:
            print("    → Credentials không hợp lệ hoặc app bị revoke")
        return

    token_data = auth_response.json()
    token      = token_data["access_token"]
    expires_in = token_data["expires_in"]
    print(f"    ✅ Xác thực THÀNH CÔNG")
    print(f"    Token: {token[:20]}...")
    print(f"    Hết hạn sau: {expires_in}s ({expires_in//60} phút)")

    # ── Bước 2: Test Search API ───────────────────────────────────
    print("\n[2] Đang test tìm kiếm bài hát...")
    search_response = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": "Shape of You", "type": "track", "limit": 1},
        timeout=10
    )

    if search_response.status_code == 429:
        retry_after = search_response.headers.get("Retry-After", "?")
        print(f"    ⚠️  Rate Limited (429)")
        print(f"    Retry-After: {retry_after}s")
        if int(retry_after) > 3600:
            print(f"    → App này đã bị dùng quá nhiều, cần chờ {int(retry_after)//3600} giờ")
        else:
            print(f"    → Thử lại sau {retry_after} giây")
    elif search_response.status_code == 200:
        items = search_response.json().get("tracks", {}).get("items", [])
        if items:
            track = items[0]
            print(f"    ✅ Search HOẠT ĐỘNG")
            print(f"    Bài hát: {track['name']}")
            print(f"    Nghệ sĩ: {track['artists'][0]['name']}")
            print(f"    Phổ biến: {track['popularity']}/100")
        else:
            print("    ⚠️  Search trả về kết quả rỗng")
    else:
        print(f"    ❌ Search thất bại: HTTP {search_response.status_code}")
        print(f"    {search_response.text[:200]}")

    # ── Bước 3: Test Audio Features API ──────────────────────────
    # (Deprecated từ 11/2024 với app mới — chỉ app cũ còn dùng được)
    print("\n[3] Đang test Audio Features API...")
    # Track ID của "Shape of You" - Ed Sheeran
    SHAPE_OF_YOU_ID = "7qiZfU4dY1lWllzX7mPBI3"
    feat_response = requests.get(
        f"https://api.spotify.com/v1/audio-features/{SHAPE_OF_YOU_ID}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10
    )

    if feat_response.status_code == 200:
        f = feat_response.json()
        print(f"    ✅ Audio Features HOẠT ĐỘNG (app này được tạo trước 11/2024)")
        print(f"    energy={f.get('energy')}, valence={f.get('valence')}, tempo={f.get('tempo'):.0f}BPM")
    elif feat_response.status_code == 403:
        print(f"    ⚠️  Audio Features: 403 Forbidden")
        print(f"    → Bình thường! Endpoint này deprecated với app tạo sau tháng 11/2024")
        print(f"    → Search và Track info vẫn hoạt động bình thường")
    elif feat_response.status_code == 429:
        print(f"    ⚠️  Rate Limited — cần chờ theo kết quả bước 2")
    else:
        print(f"    ❌ HTTP {feat_response.status_code}: {feat_response.text[:100]}")

    # ── Kết luận ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Tóm tắt:")
    if search_response.status_code == 200:
        print("  ✅ Credentials OK — app hoạt động đầy đủ")
    elif search_response.status_code == 429:
        print("  ⚠️  Credentials OK — nhưng đang bị rate limit tạm thời")
        print("  ✅ Có thể dùng sau khi hết thời gian chờ")
    else:
        print("  ❌ Credentials hoặc app có vấn đề")
    print("=" * 60)


if __name__ == "__main__":
    check_spotify(CLIENT_ID, CLIENT_SECRET)
