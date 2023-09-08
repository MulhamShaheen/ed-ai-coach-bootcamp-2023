from __future__ import annotations
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from typing import List, Optional, Callable, Iterable, Dict, Tuple
import codecs
import jinja2
import json
import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass


@dataclass
class ReportFileStructure:
    root_folder: str

    def __post_init__(self) -> None:
        os.makedirs(self.materials_root, exist_ok=True)

    @property
    def materials_root(self) -> str:
        return os.path.join(self.root_folder, "materials")

    @property
    def generated_report_path(self) -> str:
        return os.path.join(self.root_folder, "report.html")

    @property
    def radar_chart_path(self) -> str:
        return os.path.join(self.materials_root, "radar_chart.png")

    @property
    def html_table_path(self) -> str:
        return os.path.join(self.materials_root, "report_metrics.html")

    @property
    def speech_analysis_plot_path(self) -> str:
        return os.path.join(self.materials_root, "speech_analysis.png")

    @property
    def frequent_parasites_plot_path(self) -> str:
        return os.path.join(self.materials_root, "frequent_parasites.png")

    @property
    def frequent_words_plot_path(self) -> str:
        return os.path.join(self.materials_root, "frequent_words.png")


@dataclass
class PipelineOutputFileStructure:
    root_folder: str

    @property
    def speech_analysis_output_path(self):
        return os.path.join(self.root_folder, "speech_analysis.json")


class CriteriaGrade(Enum):
    bad = auto()
    medium = auto()
    good = auto()


class LectureCriteria(Enum):
    articulation_time_percentage = auto()
    clear_speech_count_percentage = auto()
    gaze_to_camera_time_percentage = auto()
    background_quality_time_percentage = auto()
    head_visibility_time_percentage = auto()
    positioned_at_frame_center_time_percentage = auto()


WORDS_STATS_TYPE = Dict
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEECH_DATA_JSON = os.path.join(TEMPLATE_DIR, "speech_data.json")
CRITERIA_GRADES_COLORS: Dict[CriteriaGrade, str] = {
    CriteriaGrade.good: "#AAFF00",
    CriteriaGrade.medium: "#FFD700",
    CriteriaGrade.bad: "#C70039",
}
CRITERIAS_GRADING_RULES: Dict[
    LectureCriteria, Dict[CriteriaGrade, Tuple[float, float]]
] = {
    LectureCriteria.articulation_time_percentage: {
        CriteriaGrade.good: (0.4, 1.1),
        CriteriaGrade.medium: (0.3, 0.4),
        CriteriaGrade.bad: (0, 0.3),
    },
    LectureCriteria.clear_speech_count_percentage: {
        CriteriaGrade.good: (0.4, 1.1),
        CriteriaGrade.medium: (0.3, 0.4),
        CriteriaGrade.bad: (0, 0.3),
    },
    LectureCriteria.gaze_to_camera_time_percentage: {
        CriteriaGrade.good: (0.4, 1.1),
        CriteriaGrade.medium: (0.3, 0.4),
        CriteriaGrade.bad: (0, 0.3),
    },
    LectureCriteria.background_quality_time_percentage: {
        CriteriaGrade.good: (0.4, 1.1),
        CriteriaGrade.medium: (0.3, 0.4),
        CriteriaGrade.bad: (0, 0.3),
    },
    LectureCriteria.head_visibility_time_percentage: {
        CriteriaGrade.good: (0.4, 1.1),
        CriteriaGrade.medium: (0.3, 0.4),
        CriteriaGrade.bad: (0, 0.3),
    },
    LectureCriteria.positioned_at_frame_center_time_percentage: {
        CriteriaGrade.good: (0.4, 1.1),
        CriteriaGrade.medium: (0.3, 0.4),
        CriteriaGrade.bad: (0, 0.3),
    },
}


def write_file(output_path: str, data: str) -> None:
    with open(output_path, "w") as f:
        f.write(data)


def test_table() -> pd.DataFrame:
    return pd.DataFrame(
        data=np.random.random((2, 2)),
        columns=["06.09.2023", "07.09.2023"],
        index=["articulation", "filler_words_usage"],
    )


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def safe_reading(path: str) -> str:
    try:
        with codecs.open(path, "r") as f:
            result = f.read()
    except FileNotFoundError:
        return ""
    return result


def plot_data_as_radar_chart(
    data: List[float], labels: List[str], color: str
) -> Figure:
    if len(data) != len(labels):
        raise ValueError("Each data point should correspond to 1 label")
    theta = radar_factory(len(data), frame="circle")
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(projection="radar"))
    ax.plot(theta, data, color=color)
    ax.fill(theta, data, facecolor=color, alpha=0.25, label="_nolegend_")
    ax.set_varlabels(labels)
    return fig


class SpeechStatisticsProcessor:
    @staticmethod
    def extract_parasites_count(data: WORDS_STATS_TYPE) -> int:
        parasites_stats = data["output"]["top parasites"]
        parasites_count = sum([count for word, count in parasites_stats])
        return parasites_count

    @staticmethod
    def extract_clear_speech_percentage(
        speech_analysis_data: WORDS_STATS_TYPE,
    ) -> float:
        parasite_count = SpeechStatisticsProcessor.extract_parasites_count(
            speech_analysis_data
        )
        overall_words_count = speech_analysis_data["output"]["overall count"]
        amm_count = speech_analysis_data["output"]["ammm count"]
        clear_words_count = overall_words_count - amm_count - parasite_count
        clear_speech_percentage = clear_words_count / overall_words_count
        return clear_speech_percentage

    @staticmethod
    def plot_speech_analysis_hbar(ax: Axes, data: WORDS_STATS_TYPE) -> None:
        parasite_count = SpeechStatisticsProcessor.extract_parasites_count(data)
        all_words_num = data["output"]["overall count"]
        amm_count = data["output"]["ammm count"]
        normal_words_count = all_words_num - parasite_count - amm_count
        normal_words_percentage = normal_words_count / all_words_num
        parasite_percentage = parasite_count / all_words_num
        amm_percentage = amm_count / all_words_num

        words_kinds_percentages = ("Words", "Parasites", "*", "")
        y_pos = np.arange(len(words_kinds_percentages))
        words_stats = [normal_words_percentage, parasite_percentage, amm_percentage, 1]
        ax.barh(
            y_pos,
            words_stats,
            align="center",
            color=[
                "g",
                "r",
                "r",
                "w",
            ],
        )
        ax.set_yticks(y_pos, labels=words_kinds_percentages)
        ax.invert_yaxis()
        ax.set_xlabel("Speech percentage")
        ax.set_title("Speech analysis")

    @staticmethod
    def plot_frequent_parasites_hbar(ax: Axes, data: WORDS_STATS_TYPE) -> None:
        parasites_count = data["output"]["top parasites"]
        parasites_counts = [count for _, count in parasites_count if count > 0]
        parasites_names = [name for name, count in parasites_count if count > 0]
        y_pos = np.arange(len(parasites_counts))
        ax.barh(y_pos, parasites_counts, align="center", color="r")
        ax.set_yticks(y_pos, labels=parasites_names)
        ax.set_xticks(list(range(max(parasites_counts) + 1)))
        ax.invert_yaxis()
        ax.set_xlabel("Usage number")
        ax.set_title("Frequent words analysis")

    @staticmethod
    def plot_frequent_words_hbar(ax: Axes, data: WORDS_STATS_TYPE) -> None:
        top_words_count = data["output"]["top words"]
        top_words_counts = [
            count for name, count in top_words_count if count > 0 and name != ""
        ]
        top_words_names = [
            name for name, count in top_words_count if count > 0 and name != ""
        ]
        y_pos = np.arange(len(top_words_counts))
        ax.barh(y_pos, top_words_counts, align="center")
        ax.set_yticks(y_pos, labels=top_words_names)
        ax.set_xticks(list(range(max(top_words_counts) + 1)))
        ax.invert_yaxis()
        ax.set_xlabel("Usage number")
        ax.set_title("Frequent words analysis")


class LectureCriteriaComputers:
    @staticmethod
    def get_articulation_percentage(
        pipeline_output_structure: PipelineOutputFileStructure,
    ) -> float:
        return np.random.random()

    @staticmethod
    def get_clear_speech_percentage(
        pipeline_output_structure: PipelineOutputFileStructure,
    ) -> float:
        speech_analysis_data: WORDS_STATS_TYPE
        with open(pipeline_output_structure.speech_analysis_output_path, "r") as f:
            speech_analysis_data = json.load(f)
        clear_speech_percentage: float = (
            SpeechStatisticsProcessor.extract_clear_speech_percentage(
                speech_analysis_data
            )
        )
        return clear_speech_percentage

    @staticmethod
    def get_gaze_to_camera_percentage(
        pipeline_output_structure: PipelineOutputFileStructure,
    ) -> float:
        return np.random.random()

    @staticmethod
    def get_background_quality_time_percentage(
        pipeline_output_structure: PipelineOutputFileStructure,
    ) -> float:
        return np.random.random()

    @staticmethod
    def get_head_visibility_time_percentage(
        pipeline_output_structure: PipelineOutputFileStructure,
    ) -> float:
        return np.random.random()

    @staticmethod
    def get_position_at_frame_center_time_percentage(
        pipeline_output_structure: PipelineOutputFileStructure,
    ) -> float:
        return np.random.random()


class ReportGeneration:
    CRITERA_ANALYZERS: Dict[
        LectureCriteria, Callable[[PipelineOutputFileStructure], float]
    ] = {
        LectureCriteria.articulation_time_percentage: LectureCriteriaComputers.get_articulation_percentage,
        LectureCriteria.clear_speech_count_percentage: LectureCriteriaComputers.get_clear_speech_percentage,
        LectureCriteria.gaze_to_camera_time_percentage: LectureCriteriaComputers.get_gaze_to_camera_percentage,
        LectureCriteria.background_quality_time_percentage: LectureCriteriaComputers.get_background_quality_time_percentage,
        LectureCriteria.head_visibility_time_percentage: LectureCriteriaComputers.get_head_visibility_time_percentage,
        LectureCriteria.positioned_at_frame_center_time_percentage: LectureCriteriaComputers.get_position_at_frame_center_time_percentage,
    }

    @staticmethod
    def determine_grade_for_value(
        grading_rule: Dict[CriteriaGrade, Tuple[float, float]], value: float
    ) -> CriteriaGrade:
        for grade, (interval_start, interval_end) in grading_rule.items():
            if interval_start <= value < interval_end:
                return grade

    @staticmethod
    def get_grades_for_table(table: pd.DataFrame) -> pd.DataFrame:
        data = table.values
        index = table.index.to_list()
        grades_data = []
        for index_name, data_row in zip(index, data):
            criteria = LectureCriteria[index_name]
            criteria_grading_rule = CRITERIAS_GRADING_RULES[criteria]
            data_row_grades: List[CriteriaGrade] = [
                ReportGeneration.determine_grade_for_value(
                    grading_rule=criteria_grading_rule, value=v
                )
                for v in data_row
            ]
            data_row_grades_names = [grade.name for grade in data_row_grades]
            grades_data.append(data_row_grades_names)
        grades_datagrame = pd.DataFrame(
            data=grades_data, index=table.index, columns=table.columns
        )
        return grades_datagrame

    @staticmethod
    def get_html_repr_of_metrics_table(metrics_dataframe: pd.DataFrame) -> str:
        cell_grades = ReportGeneration.get_grades_for_table(metrics_dataframe)
        metrics_dataframe = metrics_dataframe.applymap(lambda x: f"{x:.2f}")

        headers = {
            "selector": "th:not(.index_name)",
            "props": "background-color: #C8C8C8; color: black;",
        }
        table_attributes = 'style="border-collapse: collapse"'

        styler = (
            metrics_dataframe.style.set_table_attributes(table_attributes)
            .set_table_styles([headers])
            .set_table_styles(
                [
                    {"selector": "th", "props": 'style="border: 1px solid #181818"'},
                    {"selector": "td", "props": "border: 1px solid #181818"},
                    {"selector": "tr", "props": "border: 1px solid #181818"},
                ],
                overwrite=False,
            )
        )
        styler.set_table_styles(
            [
                {
                    "selector": f".{grade.name}",
                    "props": f"background-color: {CRITERIA_GRADES_COLORS[grade]};",
                }
                for grade in CriteriaGrade
            ],
            overwrite=False,
        )

        styler.set_td_classes(cell_grades)
        html_render = styler.to_html()
        return html_render

    @staticmethod
    def produce_metrics_dataframe(
        pipeline_output: PipelineOutputFileStructure,
        lecture_name: Optional[str] = None,
    ) -> pd.DataFrame:
        metrics_table_data = []
        row_names: List[str] = []
        for criteria, criteria_function in ReportGeneration.CRITERA_ANALYZERS.items():
            criteria_value = criteria_function(pipeline_output)
            row_names.append(criteria.name)
            metrics_table_data.append([criteria_value])

        metrics_table: pd.DataFrame
        if lecture_name:
            metrics_table = pd.DataFrame(
                metrics_table_data, index=row_names, columns=[lecture_name]
            )
        else:
            metrics_table = pd.DataFrame(
                metrics_table_data,
                index=row_names,
            )
        return metrics_table

    @staticmethod
    def generate_report_materials(
        report_structure: ReportFileStructure,
        pipeline_output_structure: PipelineOutputFileStructure,
        lecture_name: str,
    ):
        with open(pipeline_output_structure.speech_analysis_output_path, "r") as f:
            speech_data: WORDS_STATS_TYPE = json.load(f)
        metrics_dataframe: pd.DataFrame = ReportGeneration.produce_metrics_dataframe(
            pipeline_output=pipeline_output_structure, lecture_name=lecture_name
        )
        fig = plot_data_as_radar_chart(
            data=metrics_dataframe.to_numpy()[:, 0],
            labels=metrics_dataframe.index.to_list(),
            color="y",
        )
        fig.savefig(report_structure.radar_chart_path)
        plt.close(fig)

        metrics_html: str = ReportGeneration.get_html_repr_of_metrics_table(
            metrics_dataframe
        )
        write_file(output_path=report_structure.html_table_path, data=metrics_html)

        figsize = (15, 5)
        fig, ax = plt.subplots(figsize=figsize)
        SpeechStatisticsProcessor.plot_speech_analysis_hbar(ax, speech_data)
        fig.savefig(report_structure.speech_analysis_plot_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)
        SpeechStatisticsProcessor.plot_frequent_parasites_hbar(ax, speech_data)
        fig.savefig(report_structure.frequent_parasites_plot_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)
        SpeechStatisticsProcessor.plot_frequent_words_hbar(ax, speech_data)
        fig.savefig(report_structure.frequent_words_plot_path)
        plt.close(fig)

    @staticmethod
    def generate_final_report(report_structure: ReportFileStructure):
        table_summary = safe_reading(report_structure.html_table_path)

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=TEMPLATE_DIR)
        )
        template = env.get_template("report_template.html")
        radar_chart_img = f'<img src="{report_structure.radar_chart_path}"/>'
        speech_analysis_img = (
            f'<img src="{report_structure.speech_analysis_plot_path}"/>'
        )
        frequent_words_img = f'<img src="{report_structure.frequent_words_plot_path}"/>'
        frequent_parasites_img = (
            f'<img src="{report_structure.frequent_parasites_plot_path}"/>'
        )

        html = template.render(
            table_summary=table_summary,
            radar_chart=radar_chart_img,
            speech_analysis=speech_analysis_img,
            frequent_words=frequent_words_img,
            frequent_parasites=frequent_parasites_img,
        )
        write_file(output_path=report_structure.generated_report_path, data=html)


def test_table_summary():
    pipeline_output = PipelineOutputFileStructure(
        "/home/andrey/AS/dev/BootCampHak/test_pipelines/pipeline1"
    )
    metrics_dataframe = ReportGeneration.produce_metrics_dataframe(
        pipeline_output, lecture_name="08.09.2023"
    )
    print(metrics_dataframe)


if __name__ == "__main__":
    pipeline_output = PipelineOutputFileStructure(
        "/home/andrey/AS/dev/BootCampHak/test_pipelines/pipeline1"
    )
    report_structure = ReportFileStructure(
        "/home/andrey/AS/dev/BootCampHak/test_reports/report1"
    )
    ReportGeneration.generate_report_materials(
        report_structure=report_structure,
        pipeline_output_structure=pipeline_output,
        lecture_name="test",
    )
    ReportGeneration.generate_final_report(report_structure)
