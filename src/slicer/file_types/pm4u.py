import numpy as np
import os
import json
import struct
import shutil
from PIL import Image, ImageDraw, ImageFont  # Add this import for drawing text
from src.slicer.core.printer import Printer
from src.slicer.core.resin import Resin
from src.slicer.core.file_type import FileType, FileTypeRegistry


RLE_EXTENDED_LIMIT = 0xFFF  # Maximum run length for extended encoding (12 bits)


class Pm4uFile(FileType):
    type_name: str = "pm4u"
    def __init__(self, file_name: str, printer: Printer, resin: Resin, layers: list[Image.Image], intersection_levels: list[float], layer_heights: list[float], volume: float):
        self.file_name: str = file_name
        self._printer: Printer = printer
        self._resin: Resin = resin
        self._layers: list[Image.Image] = layers + [Image.new("L", printer.settings.screen_resolution, color=0)]          # Add a black dummy layer at the end (current firmware skips the last layer)
        self._intersection_levels: list[float] = intersection_levels + [intersection_levels[-1] + layer_heights[-1]]    # Add a dummy layer height at the end (current firmware skips the last layer)
        self._layer_heights: list[float] = layer_heights + [layer_heights[-1]]                                          # Add a dummy layer height at the end (current firmware skips the last layer)
        self._volume: float = volume
        self.layer_images = [self._encode_pw0(np.array(layer).flatten()) for layer in self._layers]
        self.preview_images = self._create_preview_image()
        self.print_info_file = self._create_print_info_file()
        self.layers_controller_file = self._create_layers_controller_file()
        self.software_info_file = self._create_software_info_file()
        self.scene_slice_file = self._create_scene_file()
        self.anycubic_photon_resins_file = self._create_anycubic_photon_resins_file()


    def _encode_pw0(self, image: np.ndarray) -> bytes:
        """
        Encodes a flat numpy array of dtype=uint8 using the pw0Img RLE format.
        This version computes run boundaries with NumPy.
        """

        def _flush_run(last_color: int, run_length: int) -> list:
            """
            Flushes a run (of identical quantized pixel values) into byte list.
            
            :param last_color: The quantized color (0-15).
            :param run_length: The length of the run.
            :return: A list of bytes representing the run.
            """
            b_list = []
            if last_color in (0, 15):
                # For black/white, use extended encoding with a 12-bit run length
                while run_length > 0:
                    run = min(run_length, RLE_EXTENDED_LIMIT)
                    # 16-bit value: [color (4 bits) | run length (12 bits)]
                    val = ((last_color & 0xF) << 12) | run
                    b_list.append((val >> 8) & 0xFF)  # High byte
                    b_list.append(val & 0xFF)         # Low byte
                    run_length -= run
            else:
                # For all other colors, use simple encoding with a 4-bit run length
                while run_length > 0:
                    run = min(run_length, 0xF)
                    # 8-bit value: [color (4 bits) | run length (4 bits)]
                    b_list.append((last_color << 4) | run)
                    run_length -= run
            return b_list

        # Quantize the image to 4-bit values (because the format uses 4-bit color codes)
        quantized = (image.astype(np.uint16) >> 4)
        
        # Find where the quantized value changes
        diff = np.diff(quantized)
        run_boundaries = np.nonzero(diff)[0] + 1  # indices where value changes

        # Add start and end boundaries
        boundaries = np.concatenate(([0], run_boundaries, [len(quantized)]))
        
        encoded = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            run = end - start
            value = int(quantized[start])
            encoded.extend(_flush_run(value, run))

        return bytes(encoded)


    def _create_preview_image(self) -> list[Image.Image]:
        """
        Creates a preview image with black background and white text displaying the name and layer count.
        """
        # angled view: 224x168
        # side view: 336x252
        preview_image_angled = Image.new("RGB", (224, 168), (0, 0, 0))
        preview_image_side = Image.new("RGB", (336, 252), (0, 0, 0)) 
        draw_angled = ImageDraw.Draw(preview_image_angled)
        draw_side = ImageDraw.Draw(preview_image_side)

        try:
            h1_font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype("DejaVuSans.ttf", 30)
            h2_font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype("DejaVuSans.ttf", 20)
        except IOError:
            print("DejaVuSans font not found, using default font.")
            h1_font = ImageFont.load_default()
            h2_font = ImageFont.load_default()

        # Add text to angled preview
        draw_angled.text((112, 60), self.file_name, fill="white", font=h1_font, anchor="mm")
        draw_angled.text((112, 100), f"Layers: {len(self._layers)}", fill="white", font=h2_font, anchor="mm")
        # Add text to side preview
        draw_side.text((168, 100), self.file_name, fill="white", font=h1_font, anchor="mm")
        draw_side.text((168, 140), f"Layers: {len(self._layers)}", fill="white", font=h2_font, anchor="mm")

        return [preview_image_angled, preview_image_side]


    def _create_print_info_file(self) -> dict:
        """
        Creates a json file with the information about 'cost', 'currency' , 'time', and 'volume' of the print.
        """
        exp_times_combined = 0.0
        for i, layer_height in enumerate(self._layer_heights):
            exp_times_combined += self._resin.get_layer_time(i, layer_height)

        return {
            "cost": self._resin.properties.price / 1000 * self._volume,
            "currency": self._resin.properties.currency,
            "print_time": exp_times_combined,
            "volume": self._volume
        }
        

    def _create_layers_controller_file(self) -> dict:
        """
        Creates a json file with the information about the layers and their corresponding files.
        """
        paras = []
        prev_exp_time = self._resin.get_exp_time(0, self._layer_heights[0])
        # Note: bottom settings get overwritten by the anycubic_photon_resins_file
        for i in range(len(self._layers)):
            exp_time = self._resin.get_exp_time(i, self._layer_heights[i])
            exp_time = round(exp_time + (prev_exp_time - exp_time)/3, 2) if prev_exp_time >= exp_time * 2 else exp_time
            paras.append({
                "exposure_time": exp_time,
                "layer_index": i,
                "layer_minheight": self._intersection_levels[i],
                "layer_thickness": self._layer_heights[i],
                "zup_height": self._resin.movement_settings.normal_lift_height,
                "zup_speed": self._resin.movement_settings.normal_lift_speed
            })
            prev_exp_time = exp_time

        return {
            "count": len(self._layers),
            "paras": paras
        }


    def _create_software_info_file(self) -> dict:
        """
        Creates a conf file with the information about the software used for slicing the model.
        """
        return {
            "mark": "OpenSLA",
            "opengl": "3.3-CoreProfile",
            "os": "linux-64",
            "version": "0.0.1"
        }


    def _create_scene_file(self) -> bytes:
        """
        Creates a scene.slice.
        """
        
        def _pad_string(s: str, length: int) -> bytes:
            """
            Encodes a string to ASCII and pads or truncates it to exactly 'length' bytes.
            If the string is shorter than 'length', it is null-terminated and the remaining bytes are zeros.
            """
            s_bytes = s.encode('ascii')
            if len(s_bytes) >= length:
                return s_bytes[:length]
            # Append a null terminator then pad with zeros.
            return s_bytes + b'\x00' + b'\x00' * (length - len(s_bytes) - 1)

        num_layers = len(self._layers)
        
        # Build header.
        header_fmt = "<16s64sIIIIfIffffffI64I4sI"
        magic = _pad_string("ANYCUBIC-PWSZ", 16)
        software = _pad_string("chitubox-ex", 64)
        binary_type = 3
        version = 1
        slice_type = 0
        model_unit = 0  # 0 means mm
        point_ratio = 1.0
        x_start = 0.0
        y_start = 0.0
        z_min = 0.0
        x_end = 0.0
        y_end = 0.0
        z_max = 0.0
        model_stats = 0
        padding_tuple = (0,) * 64
        separator = _pad_string("<---", 4)
        layer_def_count = num_layers
        
        header_bytes = struct.pack(
            header_fmt,
            magic,
            software,
            binary_type,
            version,
            slice_type,
            model_unit,
            point_ratio,
            num_layers,
            x_start,
            y_start,
            z_min,
            x_end,
            y_end,
            z_max,
            model_stats,
            *padding_tuple,
            separator,
            layer_def_count
        )
        
        # Build layer definitions.
        layer_fmt = "<6fIf8I"
        layers_bytes = b""
        for i in range(num_layers):
            height = self._layer_heights[i]
            area = 0.0
            lx_start = 0.0
            ly_start = 0.0
            lx_end = 0.0
            ly_end = 0.0
            object_count = 1
            max_contour_area = 0.0
            layer_padding = (0,) * 8
            
            layers_bytes += struct.pack(
                layer_fmt,
                height,
                area,
                lx_start,
                ly_start,
                lx_end,
                ly_end,
                object_count,
                max_contour_area,
                *layer_padding
            )
        
        end_marker = _pad_string("--->", 4)
        
        return header_bytes + layers_bytes + end_marker
        
        
    def _create_anycubic_photon_resins_file(self) -> dict:
        return {
            "version": "3",
            "machine_type": {
                "version": "3",
                "child_screen": [
                    {
                        "x": 0,
                        "y": 0,
                        "width": 9024,
                        "height": 5120
                    }
                ],
                "key_suffix": "pwsz",
                "key_image_format": "pwszImg",
                "max_file_version": 518,
                "max_samples": 16,
                "raster_segments_capacity": 100000,
                "raster_antialiasing": 4,
                "name": "Anycubic Photon Mono 4 Utra",
                "prev_back_color": [
                    0.0078125,
                    0.28125,
                    0.390625
                ],
                "prev_model_color": [
                    0.8046875,
                    0.8046875,
                    0.8046875
                ],
                "prev_supports_color": [
                    0.07421875,
                    0.92578125,
                    0.9296875
                ],
                "prev_image_size": [
                    224,
                    168
                ],
                "prev2_back_color": [
                    0.07842999696731567,
                    0.1058799996972084,
                    0.16077999770641327
                ],
                "prev2_image_size": [
                    336,
                    252
                ],
                "cloudprev_back_color": [
                    0.0078125,
                    0.28125,
                    0.390625
                ],
                "cloudprev_imag_size": [
                    800,
                    600
                ],
                "property": 119,
                "print_xsize": 153.408,
                "print_ysize": 87.04,
                "print_zsize": 165,
                "res_x": 9024,
                "res_y": 5120,
                "xy_pixel": 17,
                "xy_pixel_y": 17,
                "rotate_z": 180
            },
            "machine_extern": {
                "version": "3",
                "alias": "Anycubic Photon Mono M7 Pro",
                "picture": "Anycubic_Photon_Mono_M7Pro.png",
                "cloud_property": 0,
                "device_cn_code": "",
                "active_resins": [
                    "custom_resin"
                ],
                "user_resins": [
                    {
                        "version": "2",
                        "property": {
                            "version": "3",
                            "code": "10",
                            "currency": self._resin.properties.currency,
                            "name": "custom_resin",
                            "price": self._resin.properties.price,
                            "type": "Resin",
                            "volume": 1000,
                            "subfunc_code": 0,
                            "target_temperature": 25,
                            "density": self._resin.properties.density,
                        },
                        "slicepara": {
                            "anti_count": self._resin.special_settings.anti_aliasing, # TODO check which value to use
                            "blur_level": 0,
                            "gray_level": 0,    # TODO
                            "bott_layers": 1, #self._resin.resin_settings.bottom_layers,
                            "bott_time": self._resin.get_exp_time(0, self._layer_heights[0]),
                            "exposure_time": self._resin.get_exp_time(0, self._layer_heights[0]),
                            "off_time": self._resin.exposure_settings.light_off_delay,
                            "use_indivi_layerpara": 1, #0 if np.ptp(self._layer_heights) == 0 else 1, # enables individual layer parameter file
                            "use_random_erode": 0,
                            "zdown_speed": self._resin.movement_settings.normal_retract_speed,
                            "zthick": self._layer_heights[0],
                            "zup_height": self._resin.movement_settings.normal_lift_height,
                            "zup_speed": self._resin.movement_settings.normal_lift_speed
                        },
                        "slice_extpara": {
                            "version": "3",
                            "multi_state_used": 0, # if enabled indivi_layerpara does not work any more #1 if self._resin.movement_settings.use_TSMC else 0,
                            "transition_layercount": 0, #self._resin.resin_settings.transition_layers,
                            "transition_type": 0,
                            "intelli_mode": 1 if self._resin.special_settings.intelli_mode else 0, # TODO find out what this is
                            "max_acceleration": self._resin.movement_settings.max_acceleration,
                            "exposure_compensate": 0,
                            "separate_support_exposure_delayed": 0,
                            "multi_state_paras": {
                                "bott_0": {
                                    "down_speed": self._resin.movement_settings.bottom_retract_speed,
                                    "height": self._resin.movement_settings.bottom_lift_height / 2,
                                    "up_speed": self._resin.movement_settings.bottom_lift_speed
                                },
                                "bott_1": {
                                    "down_speed": self._resin.movement_settings.bottom_retract_speed * 2,
                                    "height": self._resin.movement_settings.bottom_lift_height / 2,
                                    "up_speed": self._resin.movement_settings.bottom_lift_speed * 2
                                },
                                "normal_0": {
                                    "down_speed": self._resin.movement_settings.normal_retract_speed,
                                    "height": self._resin.movement_settings.normal_lift_height / 2,
                                    "up_speed": self._resin.movement_settings.normal_lift_speed
                                },
                                "normal_1": { 
                                    "down_speed": self._resin.movement_settings.normal_retract_speed * 2,
                                    "height": self._resin.movement_settings.normal_lift_height / 2,
                                    "up_speed": self._resin.movement_settings.normal_lift_speed * 2
                                }
                            }
                        },
                        "depth_penetration_curve": {
                            "zthick_min": 0.009999999776482582,
                            "zthick_max": 0.20000000298023224,
                            "light_intensity": 9000,
                            "safety_coefficient": 1.600000023841858,
                            "current_tempcurve_selector": 0,
                            "temperature_coefficients": [
                                {
                                    "temperature": 25,
                                    "x_coefficient": 197.77999877929688,
                                    "y_compensation": 1803.199951171875
                                },
                                {
                                    "temperature": 10,
                                    "x_coefficient": 184.27000427246094,
                                    "y_compensation": 1675.800048828125
                                },
                                {
                                    "temperature": 55,
                                    "x_coefficient": 166.75999450683594,
                                    "y_compensation": 1474.0999755859375
                                },
                                {
                                    "temperature": 35,
                                    "x_coefficient": 161.19000244140625,
                                    "y_compensation": 1417.300048828125
                                },
                                {
                                    "temperature": 45,
                                    "x_coefficient": 167.30999755859375,
                                    "y_compensation": 1480.9000244140625
                                }
                            ]
                        }
                    }
                ],
                "factory_resins": [],
                "firmware_calc_print_time": 1,
                "firmware_calc_print_time_paras": {
                    "version": "2",
                    "MACHINE_AXIS_STEPS_PER_UNIT": [
                        100,
                        100,
                        3200,
                        94
                    ],
                    "MACHINE_BLOCK_BUFFER_SIZE": 32,
                    "MACHINE_DEFAULT_ACCELERATION": 1000,
                    "MACHINE_DEFAULT_MINSEGMENTTIME": 20000,
                    "MACHINE_DEFAULT_XYJERK": 20,
                    "MACHINE_DEFAULT_ZJERK": 0.20000000298023224,
                    "MACHINE_GENERATE_FRAME_TIME": 450,
                    "MACHINE_MAX_ACCELERATION": [ # TODO
                        1000,
                        1000,
                        160,
                        1000
                    ],
                    "MACHINE_MAX_FEEDRATE": [
                        200,
                        200,
                        20,
                        45
                    ],
                    "MACHINE_MAX_STEP_FREQUENCY": 256000,
                    "MACHINE_MINIMUM_PLANNER_SPEED": 0.05000000074505806,
                    "MACHINE_NOR_LAYER_DOWN_HEIGHT_DIV": 0.25,
                    "MACHINE_NOR_LAYER_DOWN_SPEED_DIV": 0.5,
                    "MACHINE_NOR_LAYER_UP_HEIGHT_DIV": 0.25,
                    "MACHINE_NOR_LAYER_UP_SPEED_DIV": 0.5,
                    "MACHINE_STEP_MUL": 1,
                    "MACHINE_TIME_COMPENSATE": 0,
                    "MACHINE_TIM_PRES": 30,
                    "MACHINE_TIM_RCC_CLK": 60,
                    "FUNCTION": 1,
                    "MACHINE_MODE_ACCELERATION": [
                        0,
                        0,
                        0,
                        0
                    ],
                    "LAYER_COMPENSATE": [
                        0,
                        0,
                        0,
                        0
                    ],
                    "HEIGHT_COMPENSATE": [
                        0,
                        0,
                        0,
                        0
                    ],
                    "TIMES_COMPENSATE": [
                        0,
                        0,
                        0,
                        0
                    ]
                },
                "firmware_calc_exp_time_paras": {
                    "precision_range_branch": [
                        0,
                        5,
                        25
                    ],
                    "precision_per_volume": 5,
                    "precision_coeff_value": [
                        0.024000000208616257,
                        0.009999999776482582,
                        -0.20000000298023224
                    ],
                    "energy_coeff": 0,
                    "machine_exposure_ton": 0.4000000059604645
                }
            }
        }


    def save(self, save_path: str) -> bool:
        """
        Saves the pm4u file to the given path.

        Parameters
        ----------
        save_path : str
            the path to save the pm4u file
        """
        path = os.path.join(os.path.dirname(__file__), "../build")
        layers_path = os.path.join(path, "layer_images")
        preview_path = os.path.join(path, "preview_images")
        save_path = os.path.join(save_path, f"{self.file_name}.pm4u")

        try:
            # clear build folder
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
            # create folders
            os.makedirs(path, exist_ok=True)
            os.makedirs(layers_path, exist_ok=True)
            os.makedirs(preview_path, exist_ok=True)
            # save files
            with open(os.path.join(path, "print_info.json"), 'w') as f:
                json.dump(self.print_info_file, f, indent=4)
            with open(os.path.join(path, "layers_controller.conf"), 'w') as f:
                json.dump(self.layers_controller_file, f, indent=4)
            with open(os.path.join(path, "software_info.conf"), 'w') as f:
                json.dump(self.software_info_file, f, indent=4)
            with open(os.path.join(path, "scene.slice"), 'wb') as f:
                f.write(self.scene_slice_file)
            with open(os.path.join(path, "anycubic_photon_resins.pwsp"), 'w') as f:
                json.dump(self.anycubic_photon_resins_file, f, indent=4)
            for i, layer in enumerate(self.layer_images):
                with open(os.path.join(layers_path, f"layer_{i}.pw0Img"), 'wb') as f:
                    f.write(layer)
            for i, preview in enumerate(self.preview_images):
                preview.save(os.path.join(preview_path, f"preview_{i}.png"))
            # zip the folder
            shutil.make_archive(self.file_name, 'zip', path)
            #rename file to have the correct extension and move it to the save path
            shutil.move(f"{self.file_name}.zip", save_path)
            return True

        except Exception as e:
            print(e)
            return False


# Register the Pm4uFile class in the FileTypeRegistry
FileTypeRegistry.register(Pm4uFile)