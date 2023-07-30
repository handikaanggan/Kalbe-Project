select married.average_age_married, single.average_age_single
from

(select avg (age) as average_age_married
from customer c 
where "Marital Status" in ('Married')) as married,

(select avg (age) as average_age_single
from customer c 
where "Marital Status" in ('Single')) as single;


