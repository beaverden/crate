import types
import typing
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr
from pydantic.fields import FieldInfo
from PySide6.QtCore import QEvent, QLocale
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QWidget
from qfluentwidgets import (
    ComboBox, DoubleSpinBox, FluentIcon, LineEdit, PasswordLineEdit, PlainTextEdit, SpinBox, SwitchButton, ToolButton,
)

__all__ = ["Editor", "editor_for"]

WIDE = 320
MAX_NUMBER = 100_000


class Editor:
    """A setting's input: `widget` to place, value() and set_value() in the setting's own type, and
    on_change(callback) called on every edit. value() raises ValueError for text that doesn't parse."""

    widget: QWidget

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        raise NotImplementedError

    def on_change(self, callback) -> None:
        raise NotImplementedError


class BoolEditor(Editor):
    def __init__(self):
        self.widget = SwitchButton()
        self.widget.setOnText("On")
        self.widget.setOffText("Off")

    def value(self) -> bool:
        return self.widget.isChecked()

    def set_value(self, value: bool) -> None:
        self.widget.setChecked(bool(value))

    def on_change(self, callback) -> None:
        self.widget.checkedChanged.connect(lambda checked: callback())


class IntEditor(Editor):
    def __init__(self):
        self.widget = SpinBox()
        self.widget.setRange(0, MAX_NUMBER)

    def value(self) -> int:
        return self.widget.value()

    def set_value(self, value: int) -> None:
        self.widget.setValue(int(value))

    def on_change(self, callback) -> None:
        self.widget.valueChanged.connect(lambda value: callback())


class PlainDoubleSpinBox(DoubleSpinBox):
    """Numbers as in the settings file: 20, 0.15, 1.5. The locale's form (20,000 with three decimals
    in Romanian) read like twenty thousand."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())

    def textFromValue(self, value: float) -> str:
        return f"{value:g}"

    def valueFromText(self, text: str) -> float:
        return float(text.replace(",", "."))

    def validate(self, text: str, pos: int):
        try:
            float(text.replace(",", ".") or "0")
        except ValueError:
            return QValidator.State.Invalid, text, pos
        return QValidator.State.Acceptable, text, pos


class FloatEditor(Editor):
    def __init__(self, default: float):
        self.widget = PlainDoubleSpinBox()
        self.widget.setRange(0, MAX_NUMBER)
        self.widget.setDecimals(3)
        self.widget.setSingleStep(0.05 if default <= 1 else 1)  # fractions (0-1) or seconds

    def value(self) -> float:
        return self.widget.value()

    def set_value(self, value: float) -> None:
        self.widget.setValue(float(value))

    def on_change(self, callback) -> None:
        self.widget.valueChanged.connect(lambda value: callback())


class SecretLineEdit(PasswordLineEdit):
    """A password field whose eye button toggles the text shown or hidden on each click, rather than
    showing it only while the button is held down."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_icon()

    def setPasswordVisible(self, visible: bool) -> None:
        super().setPasswordVisible(visible)
        self._show_icon()

    def eventFilter(self, obj, e) -> bool:
        if obj is self.viewButton and e.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            if e.type() == QEvent.Type.MouseButtonRelease and self.isEnabled():
                self.setPasswordVisible(not self.isPasswordVisible())
            # Skip PasswordLineEdit's show-while-pressed handling, keep the button's own.
            return LineEdit.eventFilter(self, obj, e)
        return super().eventFilter(obj, e)

    def _show_icon(self) -> None:
        visible = self.isPasswordVisible()
        self.viewButton.setIcon(FluentIcon.HIDE if visible else FluentIcon.VIEW)
        self.viewButton.setToolTip("Hide" if visible else "Show")


class TextEditor(Editor):
    def __init__(self, secret: bool = False):
        self.widget = SecretLineEdit() if secret else LineEdit()
        self.widget.setFixedWidth(WIDE)
        self.secret = secret

    def value(self) -> str | None:
        text = self.widget.text().strip()
        return (text or None) if self.secret else text

    def set_value(self, value: Any) -> None:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        self.widget.setText("" if value is None else str(value))
        if self.secret:
            self.widget.setPasswordVisible(False)  # hidden again after loading, saving or reverting

    def on_change(self, callback) -> None:
        self.widget.textChanged.connect(lambda text: callback())


class FolderEditor(Editor):
    def __init__(self):
        self.widget = QWidget()
        self.edit = LineEdit()
        self.edit.setFixedWidth(WIDE - 40)
        browse = ToolButton(FluentIcon.FOLDER)
        browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.edit)
        layout.addWidget(browse)

    def value(self) -> Path:
        return Path(self.edit.text().strip()).expanduser()

    def set_value(self, value: Path) -> None:
        self.edit.setText(str(value))

    def on_change(self, callback) -> None:
        self.edit.textChanged.connect(lambda text: callback())

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.widget, "Folder", self.edit.text())
        if folder:
            self.edit.setText(folder)


class ChoiceEditor(Editor):
    """One of `choices`, from a dropdown; with `optional`, NONE first, for None. A value that isn't one of them
    (written into the file by hand) is listed too, so showing the settings doesn't change it."""

    NONE = "none"

    def __init__(self, choices: tuple[str, ...], optional: bool = False):
        self.widget = ComboBox()
        self.widget.setFixedWidth(WIDE)
        self.choices, self.optional = choices, optional
        self.widget.addItems([self.NONE, *choices] if optional else choices)

    def value(self) -> str | None:
        text = self.widget.currentText()
        return None if self.optional and text == self.NONE else text

    def set_value(self, value: str | None) -> None:
        value = self.NONE if value is None and self.optional else value
        if self.widget.findText(value) < 0:
            self.widget.addItem(value)
        self.widget.setCurrentText(value)

    def on_change(self, callback) -> None:
        self.widget.currentTextChanged.connect(lambda text: callback())


class MappingEditor(Editor):
    """A mapping as YAML lines, e.g. `effort: low`."""

    def __init__(self):
        self.widget = PlainTextEdit()
        self.widget.setFixedSize(WIDE, 72)

    def value(self) -> dict:
        try:
            data = yaml.safe_load(self.widget.toPlainText())
        except yaml.YAMLError as e:
            raise ValueError(f"not valid YAML: {e}") from e
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("expected `key: value` lines")
        return data

    def set_value(self, value: dict) -> None:
        self.widget.setPlainText(yaml.safe_dump(value, sort_keys=False).strip() if value else "")

    def on_change(self, callback) -> None:
        self.widget.textChanged.connect(callback)


def editor_for(field: FieldInfo, choices: tuple[str, ...] | None = None) -> Editor:
    """The input for a setting of `field`'s type; with `choices`, one of them (or none, if the setting can be None)."""
    union = isinstance(field.annotation, types.UnionType)
    if choices is not None:
        return ChoiceEditor(choices, optional=union and type(None) in typing.get_args(field.annotation))
    kinds = set(typing.get_args(field.annotation)) - {type(None)} if union else {
        typing.get_origin(field.annotation) or field.annotation}
    if SecretStr in kinds:
        return TextEditor(secret=True)
    if bool in kinds:
        return BoolEditor()
    if int in kinds:
        return IntEditor()
    if float in kinds:
        return FloatEditor(field.default)
    if Path in kinds:
        return FolderEditor()
    if dict in kinds:
        return MappingEditor()
    return TextEditor()
