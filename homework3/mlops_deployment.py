from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from pathlib import Path

DeploymentSpec(
    name='mlops-deployment',
    flow_location=Path(__file__).parent / 'homework.py',
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron='0 9 15 * *',
        day_or=True,
        timezone='America/New_York'
    ),
)
