import os
from configparser import ConfigParser

class Config(ConfigParser):
    def __init__(self, config_file):
        # MỚI: Thêm một lệnh in để xác nhận đường dẫn file đang được đọc
        print(f"----- Bắt đầu đọc cấu hình từ: {os.path.abspath(config_file)} -----")
        
        raw_config = ConfigParser()
        
        # MỚI: Kiểm tra xem file có thực sự được đọc hay không
        read_ok = raw_config.read(config_file, encoding='utf-8')
        
        if not read_ok:
            print("\n!!! LỖI QUAN TRỌNG !!!")
            print(f"Không thể tìm thấy hoặc đọc file cấu hình tại đường dẫn: '{config_file}'")
            print("Hãy chắc chắn rằng file 'config.ini' tồn tại và nằm cùng cấp với file train.py.\n")
            # Thoát chương trình với một thông báo lỗi rõ ràng
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_file}")

        # MỚI: Kiểm tra xem file có nội dung hay không
        if not raw_config.sections():
            print("\n!!! LỖI QUAN TRỌNG !!!")
            print(f"File cấu hình '{config_file}' bị trống hoặc không có section nào hợp lệ (ví dụ: [Encoder]).")
            print("Vui lòng kiểm tra lại nội dung file config.ini.\n")
            raise ValueError(f"File cấu hình không hợp lệ: {config_file}")
            
        print("----- Đọc cấu hình thành công. Đang gán giá trị... -----")
        self.cast_values(raw_config)
        print("----- Hoàn tất cấu hình. -----\n")


    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                    try:
                        val = eval(value)
                    except (SyntaxError, NameError):
                        val = value # Giữ nguyên giá trị nếu eval lỗi
                else:
                    # Thử chuyển đổi kiểu dữ liệu
                    for attr in ["getint", "getfloat", "getboolean"]:
                        try:
                            val = getattr(raw_config[section], attr)(key)
                            break
                        except (ValueError, KeyError):
                            val = value
                
                # Gán thuộc tính vào self
                setattr(self, key, val)
