import click
import pandas as pd
from click import ParamType


DATE_FORMAT = "%Y-%m-%d"


class SpecialHelpOrder(click.Group):
    """Allows user to order the display of the CLI commands"""

    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super(SpecialHelpOrder, self).get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 99), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop("help_priority", 99)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


class CLITimestamp(ParamType):
    """
    A custom caster that allows "pd.Timestamp" as a type for input CLI arguments.
    """

    name = "timestamp"

    def convert(self, value, param, ctx):
        try:
            return pd.to_datetime(value, format=DATE_FORMAT)
        except ValueError:
            self.fail(f"{value} is not a valid '{DATE_FORMAT}' date", param, ctx)

    def __repr__(self):
        return "TIMESTAMP"


def make_list_cb(ctx, param, value):
    """
    A click callback to make sure the multi argument option is cast into a list
    """
    try:
        return list(value)
    except ValueError:
        raise click.BadParameter(
            f"Cannot cast CLI parameter {param} value {value} to list"
        )
